import random
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
import os
import json
import warnings
import tqdm
from torch.utils.data import DataLoader

from util import evaluate, plot_pr_curves, calculate_pr_curve_with_mapping, plot_pr_curve_from_data
from dataset import CountDataset
from model_CountMamba import CountMambaModel

warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

# Config
parser = argparse.ArgumentParser(description="WFlib")

# Dataset
parser.add_argument('--dataset', default="CW")
parser.add_argument("--load_ratio", type=int, default=100)
parser.add_argument("--result_file", type=str, default="test_p100", help="File to save test results")
parser.add_argument('--num_tabs', default=1, type=int)
parser.add_argument('--fine_predict', action="store_true")
parser.add_argument("--mapping_file", type=str, default=None, help="Optional mapping file name")
# Representation
parser.add_argument('--seq_len', default=5000, type=int)
parser.add_argument('--maximum_load_time', default=80, type=int)
parser.add_argument('--max_matrix_len', default=1800, type=int)
parser.add_argument('--time_interval_threshold', default=0.1, type=float)
parser.add_argument('--maximum_cell_number', default=2, type=int)
parser.add_argument('--log_transform', action="store_true")

# Model
parser.add_argument('--drop_path_rate', default=0.2, type=float)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--embed_dim', default=256, type=int)
# Train
parser.add_argument('--batch_size', default=200, type=int)
# Evaluation
parser.add_argument('--plot_pr_curve', action="store_true", help="Plot PR curve")

args = parser.parse_args()

in_path = os.path.join("../npz_dataset", args.dataset)
log_path = os.path.join("../logs", args.dataset, "CountMamba")
ckp_path = os.path.join("../checkpoints/", args.dataset, "CountMamba")
os.makedirs(log_path, exist_ok=True)
out_file = os.path.join(log_path, f"{args.result_file}.json")

# Construct mapping file path if provided
mapping_file = None
if args.mapping_file is not None:
    mapping_file = os.path.join(in_path, f"{args.mapping_file}.npy")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if args.fine_predict:
        BAPM = data["BAPM"]
        return X, BAPM, y
    else:
        return X, y


if args.load_ratio != 100:
    test_X, test_y = load_data(os.path.join(in_path, f"test_p{args.load_ratio}.npz"))
else:
    if args.fine_predict:
        test_X, test_BAPM, test_y = load_data(os.path.join(in_path, f"test.npz"))
    else:
        test_X, test_y = load_data(os.path.join(in_path, f"test.npz"))

if args.num_tabs == 1:
    num_classes = len(np.unique(test_y))
else:
    num_classes = test_y.shape[1]

if args.fine_predict:
    dataset_test = CountDataset(test_X, test_y, args=args, BAPM=test_BAPM)
else:
    dataset_test = CountDataset(test_X, test_y, args=args)
    
data_loader_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=20,
    pin_memory=True,
    drop_last=False
)

# Model
patch_size = 2 * (args.maximum_cell_number + 1) + 2
model = CountMambaModel(num_classes=num_classes, drop_path_rate=args.drop_path_rate, depth=args.depth,
                        embed_dim=args.embed_dim, patch_size=patch_size, max_matrix_len=args.max_matrix_len,
                        early_stage=False, num_tabs=args.num_tabs, fine_predict=args.fine_predict)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"max_f1.pth"), map_location="cpu"))
model.cuda()

# # ============================================================
# # Model Performance Analysis
# # ============================================================
# import time

# perf_metrics = {}

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# perf_metrics['params_M'] = total_params / 1e6
# perf_metrics['trainable_params_M'] = trainable_params / 1e6

# model.eval()
# for sample_data in data_loader_test:
#     sample_input = sample_data[0][0][0:1].cuda()
#     sample_idx = sample_data[0][1][0:1].cuda()
#     break

# # Custom FLOPs calculation using hooks
# def count_flops_with_hooks(model, inputs):
#     """Count FLOPs using hooks to capture layer-wise operations"""
#     total_flops = 0
    
#     def conv2d_hook(module, input, output):
#         nonlocal total_flops
#         batch_size = input[0].size(0)
#         output_height = output.size(2)
#         output_width = output.size(3)
        
#         kernel_height, kernel_width = module.kernel_size
#         in_channels = module.in_channels
#         out_channels = module.out_channels
#         groups = module.groups
        
#         # FLOPs = batch_size * output_size * kernel_ops * out_channels
#         kernel_flops = kernel_height * kernel_width * (in_channels // groups)
#         output_size = output_height * output_width
#         flops = batch_size * output_size * kernel_flops * out_channels
        
#         if module.bias is not None:
#             flops += batch_size * output_size * out_channels
        
#         total_flops += flops
    
#     def conv1d_hook(module, input, output):
#         nonlocal total_flops
#         batch_size = input[0].size(0)
#         output_length = output.size(2)
        
#         kernel_size = module.kernel_size[0]
#         in_channels = module.in_channels
#         out_channels = module.out_channels
#         groups = module.groups
        
#         kernel_flops = kernel_size * (in_channels // groups)
#         flops = batch_size * output_length * kernel_flops * out_channels
        
#         if module.bias is not None:
#             flops += batch_size * output_length * out_channels
        
#         total_flops += flops
    
#     def linear_hook(module, input, output):
#         nonlocal total_flops
#         batch_size = input[0].size(0)
#         if len(input[0].shape) == 3:
#             seq_len = input[0].size(1)
#             batch_size = batch_size * seq_len
        
#         in_features = module.in_features
#         out_features = module.out_features
        
#         # FLOPs = batch_size * in_features * out_features
#         flops = batch_size * in_features * out_features
        
#         if module.bias is not None:
#             flops += batch_size * out_features
        
#         total_flops += flops
    
#     def batchnorm_hook(module, input, output):
#         nonlocal total_flops
#         # BatchNorm: 2 * num_elements (mean and variance)
#         flops = input[0].numel() * 2
#         total_flops += flops
    
#     hooks = []
#     for module in model.modules():
#         if isinstance(module, nn.Conv2d):
#             hooks.append(module.register_forward_hook(conv2d_hook))
#         elif isinstance(module, nn.Conv1d):
#             hooks.append(module.register_forward_hook(conv1d_hook))
#         elif isinstance(module, nn.Linear):
#             hooks.append(module.register_forward_hook(linear_hook))
#         elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             hooks.append(module.register_forward_hook(batchnorm_hook))
    
#     # Forward pass
#     with torch.no_grad():
#         if isinstance(inputs, tuple):
#             _ = model(*inputs)
#         else:
#             _ = model(inputs)
    
#     # Remove hooks
#     for hook in hooks:
#         hook.remove()
    
#     return total_flops

# flops_calculated = False

# # Try custom hook-based method first (most reliable for Mamba)
# try:
#     flops = count_flops_with_hooks(model, (sample_input, sample_idx))
#     perf_metrics['flops_G'] = flops / 1e9
#     flops_calculated = True
# except Exception as e:
#     print(f"Hook-based FLOPs calculation failed: {type(e).__name__}: {e}")

# # Try thop as fallback
# if not flops_calculated:
#     try:
#         from thop import profile
#         flops, _ = profile(model, inputs=(sample_input, sample_idx), verbose=False)
#         perf_metrics['flops_G'] = flops / 1e9
#         flops_calculated = True
#     except Exception as e:
#         pass

# # If all methods fail
# if not flops_calculated:
#     perf_metrics['flops_G'] = None
#     perf_metrics['flops_note'] = "calculation not supported"

# warmup_runs = 10
# test_runs = 100

# # Get a fresh sample for inference timing
# for sample_data in data_loader_test:
#     sample_input = sample_data[0][0][0:1].cuda()
#     sample_idx = sample_data[0][1][0:1].cuda()
#     break

# with torch.no_grad():
#     for _ in range(warmup_runs):
#         _ = model(sample_input, sample_idx)
#         torch.cuda.synchronize()

# inference_times = []
# with torch.no_grad():
#     for _ in range(test_runs):
#         torch.cuda.synchronize()
#         start = time.time()
#         _ = model(sample_input, sample_idx)
#         torch.cuda.synchronize()
#         end = time.time()
#         inference_times.append((end - start) * 1000)

# perf_metrics['inference_time_ms'] = np.mean(inference_times)
# perf_metrics['inference_time_std_ms'] = np.std(inference_times)
# perf_metrics['inference_note'] = "model forward only"

# print("\nModel Performance Analysis:")
# print(f"  - FLOPs: per-sample forward pass")
# print(f"  - Inference: inference time ({perf_metrics.get('inference_note', '')})")

# print("\n" + "="*70)
# print(f"{'Model':<15} {'#Params (M)':<15} {'FLOPs (G)':<15} {'Inference (ms)':<20}")
# print("-"*70)

# model_name = "CountMamba"

# if perf_metrics['flops_G'] is not None:
#     flops_str = f"{perf_metrics['flops_G']:.2f}"
# else:
#     flops_str = "N/A"

# if perf_metrics['inference_time_ms'] is not None:
#     time_str = f"{perf_metrics['inference_time_ms']:.2f}±{perf_metrics['inference_time_std_ms']:.2f}"
# else:
#     time_str = "N/A"

# print(f"{model_name:<15} {perf_metrics['params_M']:<15.2f} {flops_str:<15} {time_str:<20}")
# print("="*70 + "\n")

# ============================================================
# Model Evaluation
# ============================================================

with torch.no_grad():
    model.eval()
    
    # Check if mapping functionality is needed
    if args.num_tabs == 1 and mapping_file is not None:
        # Collect predictions, true labels, and logits for mapping
        y_pred = []
        y_true = []
        all_logits = [] if args.plot_pr_curve else None
        
        for index, cur_data in enumerate(tqdm.tqdm(data_loader_test)):
            cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda()
            idx = cur_data[0][1].cuda()
            
            outs = model(cur_X, idx)
            
            # Save logits for PR curve
            if args.plot_pr_curve:
                all_logits.append(outs.cpu().numpy())
            
            outs = torch.argsort(outs, dim=1, descending=True)[:, 0]
            y_pred.append(outs.cpu().numpy())
            y_true.append(cur_y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        if args.plot_pr_curve:
            all_logits = np.concatenate(all_logits, axis=0)
        
        # Calculate webpage-level metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        webpage_result = {
            "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "Precision": round(precision_score(y_true, y_pred, average="macro") * 100, 2),
            "Recall": round(recall_score(y_true, y_pred, average="macro") * 100, 2),
            "F1-score": round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        }
        
        # Load mapping file and calculate website-level metrics
        mapping_array = np.load(mapping_file)
        print(f"Detected mapping file: {mapping_file}")
        
        website_y_true = mapping_array[y_true.astype(int)]
        website_y_pred = mapping_array[y_pred.astype(int)]
        
        website_result = {
            "Accuracy": round(accuracy_score(website_y_true, website_y_pred) * 100, 2),
            "Precision": round(precision_score(website_y_true, website_y_pred, average="macro") * 100, 2),
            "Recall": round(recall_score(website_y_true, website_y_pred, average="macro") * 100, 2),
            "F1-score": round(f1_score(website_y_true, website_y_pred, average="macro") * 100, 2)
        }
        
        print(f"(Webpage) {webpage_result}")
        print(f"(Website) {website_result}")
        
        # Plot PR curves if requested
        if args.plot_pr_curve and all_logits is not None:
            # Load mapping array
            mapping_array = np.load(mapping_file)
            
            # Plot webpage PR curve using original function (has logits)
            pr_curve_fig_webpage = os.path.join(log_path, 'pr_curve_webpage.png')
            pr_curve_coords_webpage = os.path.join(log_path, 'pr_curve_coords_webpage.json')
            pr_results_webpage = plot_pr_curves(y_true, all_logits, pr_curve_fig_webpage, pr_curve_coords_webpage)
            print(f"PR-Curve Results (Webpage): {pr_results_webpage}")
            
            # Calculate and plot website PR curve using mapping (no direct logits)
            _, _, website_prec, website_rec, _, _, website_auprc = \
                calculate_pr_curve_with_mapping(y_true, all_logits, mapping_array)
            
            pr_curve_fig_website = os.path.join(log_path, 'pr_curve_website.png')
            pr_curve_coords_website = os.path.join(log_path, 'pr_curve_coords_website.json')
            pr_results_website = plot_pr_curve_from_data(
                website_prec, website_rec, None, website_auprc,
                pr_curve_fig_website, pr_curve_coords_website,
                n_classes=len(np.unique(website_y_true)), level_name="Website"
            )
            print(f"PR-Curve Results (Website): {pr_results_website}")
        
        # Save hierarchical results     
        combined_result = {
            "webpage": [webpage_result, {}],
            "website": [website_result, {}]
        }
        
        with open(out_file, "w") as fp:
            json.dump(combined_result, fp, indent=4)
    else:
        # Use original evaluate function
        if args.plot_pr_curve and args.num_tabs == 1:
            # Collect logits for PR curve
            result, y_true, y_pred, all_logits = evaluate(model, data_loader_test, args.num_tabs, num_classes, args.fine_predict, return_logits=True)
            print(result)
            
            # Plot PR curve
            pr_curve_fig = os.path.join(log_path, 'pr_curve.png')
            pr_curve_coords = os.path.join(log_path, 'pr_curve_coords.json')
            pr_results = plot_pr_curves(y_true, all_logits, pr_curve_fig, pr_curve_coords)
            print(f"PR-Curve Results: {pr_results}")
        else:
            result = evaluate(model, data_loader_test, args.num_tabs, num_classes, args.fine_predict)
            print(result)

        with open(out_file, "w") as fp:
            json.dump(result, fp, indent=4)
