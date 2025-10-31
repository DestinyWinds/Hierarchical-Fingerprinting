import torch.nn as nn
import torch
from model_mamba2 import Mamba2
from timm.layers import DropPath
import numpy as np


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32)
    # print(grid.shape)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(1, 1))

    def forward(self, x):
        x = self.proj(x)
        return x


class CausalCNN(nn.Module):
    def __init__(self, in_channels, mid_channel, kernel_size=5):
        super(CausalCNN, self).__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1)
        self.bn2 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1)
        self.bn3 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1)
        self.bn4 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(2)

        x = self.conv1(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.unsqueeze(2)

        return x


class CountMambaModel(nn.Module):
    def __init__(self, num_classes, drop_path_rate, embed_dim, depth, num_patches, patch_size):
        super(CountMambaModel, self).__init__()

        self.patch_embed = PatchEmbed(patch_size=(patch_size, 1), in_chans=1, embed_dim=embed_dim)
        self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim//4)
            for i in range(depth)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def _get_conv_states_from_cache(self, inference_params):
        if "conv" not in inference_params.key_value_memory_dict:
            conv_state_1 = torch.zeros(
                1,
                self.local_model.conv1.weight.shape[0],
                self.local_model.kernel_size - 1,
                device=self.local_model.conv1.weight.device,
                dtype=self.local_model.conv1.weight.dtype,
            )
            conv_state_2 = torch.zeros(
                1,
                self.local_model.conv2.weight.shape[0],
                self.local_model.kernel_size - 1,
                device=self.local_model.conv2.weight.device,
                dtype=self.local_model.conv2.weight.dtype,
            )
            conv_state_3 = torch.zeros(
                1,
                self.local_model.conv3.weight.shape[0],
                self.local_model.kernel_size - 1,
                device=self.local_model.conv3.weight.device,
                dtype=self.local_model.conv3.weight.dtype,
            )
            conv_state_4 = torch.zeros(
                1,
                self.local_model.conv4.weight.shape[0],
                self.local_model.kernel_size - 1,
                device=self.local_model.conv4.weight.device,
                dtype=self.local_model.conv4.weight.dtype,
            )

            inference_params.key_value_memory_dict["conv"] = (conv_state_1, conv_state_2, conv_state_3, conv_state_4)

        conv_state = inference_params.key_value_memory_dict["conv"]
        return conv_state

    def forward_step(self, x, inference_params, history_feature, position_index, update):
        x = self.patch_embed(x).squeeze(2)

        # local model
        conv_state_1, conv_state_2, conv_state_3, conv_state_4 = self._get_conv_states_from_cache(inference_params)
        # Conv1
        x = torch.cat([conv_state_1, x], dim=-1)
        if update:
            conv_state_1.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
        x = self.local_model.conv1(x)
        x = self.local_model.bn1(x)
        x = self.local_model.relu1(x)

        # Conv2
        x = torch.cat([conv_state_2, x], dim=-1)
        if update:
            conv_state_2.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
        x = self.local_model.conv2(x)
        x = self.local_model.bn2(x)
        x = self.local_model.relu2(x)

        # Pool1
        x = self.local_model.pool1(x)
        x = self.local_model.dropout1(x)

        # Conv3
        x = torch.cat([conv_state_3, x], dim=-1)
        if update:
            conv_state_3.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
        x = self.local_model.conv3(x)
        x = self.local_model.bn3(x)
        x = self.local_model.relu3(x)

        # Conv4
        x = torch.cat([conv_state_4, x], dim=-1)
        if update:
            conv_state_4.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
        x = self.local_model.conv4(x)
        x = self.local_model.bn4(x)
        x = self.local_model.relu4(x)

        # Pool2
        x = self.local_model.pool2(x)
        x = self.local_model.dropout2(x)

        # pos embedding
        x = x.transpose(1, 2)

        clip_position_index = min(position_index, 299)
        x = x + self.pos_embed[:, 1 + clip_position_index, :]
        if clip_position_index == 0:
            cls_x = self.cls_token + self.pos_embed[:, :1, :]
            cls_x = cls_x.squeeze(1)

            for blk, drop in zip(self.blocks, self.droppaths):
                cls_x = drop(blk.forward_stage(cls_x, inference_params, update)) + cls_x

        # apply Transformer blocks
        for blk, drop in zip(self.blocks, self.droppaths):
            x = drop(blk.forward_stage(x, inference_params, update)) + x

        x = self.fc_norm(x)

        history_feature = (history_feature * position_index + x) / (position_index + 1)
        x = self.fc(history_feature)

        return x, history_feature
