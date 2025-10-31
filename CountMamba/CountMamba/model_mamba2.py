import torch.nn as nn
import torch
import math
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from einops import rearrange, repeat


class Mamba2(nn.Module):
    def __init__(self, layer_idx, d_model=256, d_state=128, d_conv=4, expand=2, headdim=64, chunk_size=256):
        device = "cuda"
        dtype = torch.float32
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Mamba2, self).__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = (self.expand * self.d_model)
        self.headdim = headdim
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=0.0001)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(1, 1.1)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False,
                                 group_size=self.d_inner, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward_train(self, u):
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())

        out = mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=self.chunk_size,
            seq_idx=None,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=1,
            norm_before_gate=False
        )
        return out

    def _get_states_from_cache(self, inference_params, batch_size):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        return conv_state, ssm_state

    def forward_test(self, u, inference_params):
        conv_state, ssm_state = self._get_states_from_cache(inference_params, 1)
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1
        )

        # conv state
        xBC_t = rearrange(xBC, "b l d -> b d l")
        conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))

        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
        )
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=1),
            rearrange(C, "b l (g n) -> b l g n", g=1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=None,
            cu_seqlens=None,
            return_final_states=True,
            return_varlen_states=False,
        )

        # ssm state
        y, last_state = y
        ssm_state.copy_(last_state)

        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)

        out = self.out_proj(y)
        return out

    def forward(self, u, inference_params=None):
        out = self.forward_train(u)
        return out

    def forward_evaluate(self, u, inference_params=None):
            start_position = u.shape[1]
            u0 = u[:, :start_position, :]
            out = self.forward_test(u0, inference_params)
            for i in range(start_position, u.shape[1]):
                out = self.forward_stage(u[:, i: i+1, :], inference_params)

            return out

    def forward_stage(self, hidden_states, inference_params):
        conv_state, ssm_state = self._get_states_from_cache(inference_params, 1)
        dtype = hidden_states.dtype

        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        # SSM step
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out.unsqueeze(1)
