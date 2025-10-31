import math
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import json
import os


def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1

    return binary


def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="Accuracy Precision Recall F1-score")
    return result


def process_addition_class(one_hot_tensor):
    num_ones = one_hot_tensor.sum(dim=2)

    only_one = (num_ones == 1).float()
    one_hot_tensor[..., -2] = only_one

    no_ones = (num_ones == 0).float()
    one_hot_tensor[..., -1] = no_ones
    one_hot_tensor[..., -2] += no_ones

    return one_hot_tensor


def process_one_hot(BAPM, num_classes, num_patches):
    part_length = BAPM.shape[-1] // num_patches
    one_hot = torch.zeros((BAPM.shape[0], num_patches, num_classes + 2), dtype=torch.float32)
    for i in range(num_patches):
        start_idx = i * part_length
        end_idx = start_idx + part_length

        current_part = BAPM[:, :, start_idx:end_idx]
        for cls in range(num_classes):
            cls_out = (current_part == cls).any(dim=1).float().any(dim=1).float()
            one_hot[:, i, cls] = cls_out

    one_hot = process_addition_class(one_hot)
    return one_hot


def process_BAPM(BAPM, num_classes):
    neg_ones_mask = (BAPM == -1)
    neg_ones_count = neg_ones_mask.sum(dim=1)
    BAPM[neg_ones_mask & (neg_ones_count.unsqueeze(1) == 1)] = num_classes
    indices_double_neg = neg_ones_mask & (neg_ones_count.unsqueeze(1) == 2)
    for b in range(BAPM.shape[0]):
        if indices_double_neg[b].any():
            BAPM[b, 0, indices_double_neg[b, 0]] = num_classes
            BAPM[b, 1, indices_double_neg[b, 1]] = num_classes + 1

    return BAPM


def measurement(y_true, y_pred, eval_metrics):
    eval_metrics = eval_metrics.split(" ")
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def evaluate(model, loader, num_tabs, num_classes, fine_predict, return_logits=False):
    if num_tabs > 1:
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))
        if fine_predict:
            y_pred_fine = []
            y_true_fine = []

        with torch.no_grad():
            model.eval()
            for index, cur_data in enumerate(tqdm.tqdm(loader)):
                cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda()
                idx = cur_data[0][1].cuda()

                if fine_predict:
                    outs, outs_fine, _ = model(cur_X, idx)

                    BAPM = cur_data[0][2]
                    BAPM = process_BAPM(BAPM, num_classes - 1)
                    one_hot = process_one_hot(BAPM, num_classes - 1, model.num_patches)
                    one_hot = one_hot.cuda()

                    fine_pred = torch.argsort(outs_fine, dim=-1)[:, :, -2:]
                    fine_label = torch.tensor([torch.nonzero(sample).squeeze().tolist() for sample in one_hot.view(-1, one_hot.shape[-1])])
                    fine_label = fine_label.view(-1, model.num_patches, 2)

                    sorted_fine_pred = torch.gather(fine_pred, dim=-1, index=torch.argsort(fine_pred, dim=-1))
                    sorted_fine_label = torch.gather(fine_label, dim=-1, index=torch.argsort(fine_label, dim=-1))
                    y_pred_fine.append(sorted_fine_pred.cpu().numpy())
                    y_true_fine.append(sorted_fine_label.cpu().numpy())
                else:
                    outs = model(cur_X, idx)
                y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
                y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

            max_tab = 5
            tp = {}
            for tab in range(1, max_tab + 1):
                tp[tab] = 0

            for idx in range(y_pred_score.shape[0]):
                cur_pred = y_pred_score[idx]
                for tab in range(1, max_tab + 1):
                    target_webs = cur_pred.argsort()[-tab:]
                    for target_web in target_webs:
                        if y_true[idx, target_web] > 0:
                            tp[tab] += 1
            mapk = .0
            for tab in range(1, max_tab + 1):
                p_tab = tp[tab] / (y_true.shape[0] * tab)
                mapk += p_tab
                if tab == num_tabs:
                    result = {
                        f"p@{tab}": round(p_tab, 4) * 100,
                        f"ap@{tab}": round(mapk / tab, 4) * 100
                    }

        if fine_predict:
            y_pred_fine = np.concatenate(y_pred_fine, axis=0)
            y_true_fine = np.concatenate(y_true_fine, axis=0)

            metric_result = compute_metric(y_true_fine, y_pred_fine)
            for k in list(metric_result.keys()):
                metric_result["fine_" + k] = metric_result[k]
                del metric_result[k]
            result.update(metric_result)

        y_pred_coarse = y_pred_score.argsort()[:, -2:]
        y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
        y_true_coarse = np.array(y_true_coarse)

        metric_coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
        result.update(metric_coarse_result)
        return result
    else:
        with torch.no_grad():
            model.eval()
            valid_pred = []
            valid_true = []
            all_logits = [] if return_logits else None

            for index, cur_data in enumerate(tqdm.tqdm(loader)):
                cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda()
                idx = cur_data[0][1].cuda()

                outs = model(cur_X, idx)
                
                # Save logits if requested
                if return_logits:
                    all_logits.append(outs.cpu().numpy())

                outs = torch.argsort(outs, dim=1, descending=True)[:, 0]
                valid_pred.append(outs.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())

            valid_pred = np.concatenate(valid_pred)
            valid_true = np.concatenate(valid_true)
            if return_logits:
                all_logits = np.concatenate(all_logits, axis=0)

        valid_result = measurement(valid_true, valid_pred, "Accuracy Precision Recall F1-score")
        
        if return_logits:
            return valid_result, valid_true, valid_pred, all_logits
        else:
            return valid_result


def get_layer_id_for_vit(name, num_layers):
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('PL_pos_embed'):
        return 0
    elif name.startswith('local_model'):
        return 0
    elif name.startswith('blocks') or name.startswith('PL_blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, prefix=""):
    param_groups = {}
    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = prefix + "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]

            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def plot_pr_curves(y_true, y_score, out_file_figure, out_file_json):
    """
    Plot macro-average Precision-Recall curve and save coordinates.
    
    Parameters:
    y_true (array): True labels (shape: [n_samples])
    y_score (array): Prediction scores/logits (shape: [n_samples, n_classes])
    out_file_figure (str): Path to save the PR curve figure
    out_file_json (str): Path to save the PR curve coordinates (JSON)
    
    Returns:
    dict: Dictionary with "AUPRC" key
    """
    # Convert logits to probabilities if needed
    if np.max(np.abs(y_score)) > 10:  # Likely logits
        y_score = softmax(y_score, axis=1)
    
    # Get unique classes
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=classes)
    
    # Handle binary classification case
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # Calculate PR curve for each class
    all_precisions = []
    all_thresholds = []
    all_aps = []
    
    # Use 101 points for interpolation (0.00, 0.01, ..., 1.00)
    recall_levels = np.linspace(0, 1, 101)
    
    for i, cls in enumerate(classes):
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        all_aps.append(ap)
        
        # Interpolate precision to common recall levels (reverse order for proper interpolation)
        precision_interp = np.interp(recall_levels, recall[::-1], precision[::-1], left=0, right=1)
        all_precisions.append(precision_interp)
        
        # Interpolate thresholds
        if len(thresholds) > 0:
            thresholds_extended = np.append(thresholds, 0)
            threshold_interp = np.interp(recall_levels, recall[::-1], thresholds_extended[::-1], left=1, right=0)
            all_thresholds.append(threshold_interp)
        else:
            all_thresholds.append(np.zeros_like(recall_levels))
    
    # Calculate macro-average
    macro_precision = np.mean(all_precisions, axis=0)
    macro_threshold = np.mean(all_thresholds, axis=0)
    macro_auprc = np.mean(all_aps) * 100
    
    # Prepare result and save data
    results = {"AUPRC": round(macro_auprc, 2)}
    
    pr_data = {
        "recall": recall_levels.tolist(),
        "precision": macro_precision.tolist(),
        "threshold": macro_threshold.tolist(),
        "auprc": round(macro_auprc, 2),
        "n_classes": n_classes
    }
    
    # Save coordinates to JSON file
    if out_file_json:
        print(f"Saving macro-average PR curve coordinates to: {out_file_json}")
        with open(out_file_json, 'w') as f:
            json.dump(pr_data, f, indent=4)
    
    # Plot macro-average PR curve
    if out_file_figure:
        print(f"Plotting macro-average PR curve to: {out_file_figure}")
        plt.figure(figsize=(10, 8))
        
        try:
            # Sample data points (every 6th point for better visualization)
            sample_step = 6
            sample_indices = np.arange(0, len(recall_levels), sample_step)
            
            # Ensure we include the last point
            if sample_indices[-1] != len(recall_levels) - 1:
                sample_indices = np.append(sample_indices, len(recall_levels) - 1)
            
            recall_sampled = recall_levels[sample_indices]
            precision_sampled = macro_precision[sample_indices]
            
            # Apply smoothing only on sampled points
            if len(precision_sampled) > 5:
                window_length = min(len(precision_sampled) if len(precision_sampled) % 2 == 1 else len(precision_sampled) - 1, 11)
                if window_length >= 5:
                    polyorder = 3
                    precision_sampled_smooth = savgol_filter(precision_sampled, window_length, polyorder)
                    # Apply additional Gaussian smoothing
                    precision_sampled_smooth = gaussian_filter1d(precision_sampled_smooth, sigma=1.5)
                    precision_sampled_smooth = np.clip(precision_sampled_smooth, 0, 1)
                else:
                    precision_sampled_smooth = precision_sampled
            else:
                precision_sampled_smooth = precision_sampled
            
            # Plot smooth curve through sampled points
            plt.plot(recall_sampled, precision_sampled_smooth, color='blue', lw=2.5, 
                    label=f'Macro-average (AUPRC={macro_auprc:.2f}%)')
            
            # Plot sample points with larger markers
            plt.scatter(recall_sampled, precision_sampled_smooth, 
                       color='blue', s=50, alpha=0.7, zorder=3, 
                       edgecolors='white', linewidths=1.0)
            
        except Exception as e:
            # Fallback to original plot if smoothing fails
            print(f"Smoothing failed: {e}, using original curve")
            plt.plot(recall_levels, macro_precision, color='blue', lw=2.5, 
                    label=f'Macro-average (AUPRC={macro_auprc:.2f}%)')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Macro-Average Precision-Recall Curve\n({n_classes} classes)', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(out_file_figure, dpi=150)
        plt.close()
    
    return results


def calculate_pr_curve_with_mapping(y_true_webpage, y_score_webpage, mapping_array):
    """
    Calculate macro-average PR curves for both webpage and website levels.
    Uses per-class PR curves and then averages, consistent with plot_pr_curves().
    
    Parameters:
    y_true_webpage (array): True webpage labels (shape: [n_samples])
    y_score_webpage (array): Webpage prediction scores/logits (shape: [n_samples, n_webpages])
    mapping_array (array): Mapping from webpage_id to website_id
    
    Returns:
    tuple: (webpage_precision, webpage_recall, website_precision, website_recall, 
            thresholds, webpage_auprc, website_auprc)
    """
    # Convert logits to probabilities
    if np.max(np.abs(y_score_webpage)) > 10:
        y_score_webpage = softmax(y_score_webpage, axis=1)
    
    # Get webpage classes
    webpage_classes = np.unique(y_true_webpage)
    n_webpages = len(webpage_classes)
    
    # Binarize webpage labels
    y_true_webpage_bin = label_binarize(y_true_webpage, classes=webpage_classes)
    if n_webpages == 2:
        y_true_webpage_bin = np.hstack([1 - y_true_webpage_bin, y_true_webpage_bin])
    
    # Map to website level
    y_true_website = mapping_array[y_true_webpage.astype(int)]
    website_classes = np.unique(y_true_website)
    n_websites = len(website_classes)
    
    # Create website-level scores by aggregating webpage scores
    y_score_website = np.zeros((len(y_true_webpage), n_websites))
    for webpage_id in range(len(mapping_array)):
        if webpage_id < y_score_webpage.shape[1]:
            website_id = mapping_array[webpage_id]
            y_score_website[:, website_id] += y_score_webpage[:, webpage_id]
    
    # Binarize website labels
    y_true_website_bin = label_binarize(y_true_website, classes=website_classes)
    if n_websites == 2:
        y_true_website_bin = np.hstack([1 - y_true_website_bin, y_true_website_bin])
    
    # Calculate PR curves for each webpage class
    webpage_precisions = []
    webpage_recalls = []
    webpage_aps = []
    recall_levels = np.linspace(0, 1, 101)
    
    for i, cls in enumerate(webpage_classes):
        precision, recall, _ = precision_recall_curve(y_true_webpage_bin[:, i], y_score_webpage[:, i])
        ap = average_precision_score(y_true_webpage_bin[:, i], y_score_webpage[:, i])
        webpage_aps.append(ap)
        
        # Interpolate to common recall levels
        precision_interp = np.interp(recall_levels, recall[::-1], precision[::-1], left=0, right=1)
        webpage_precisions.append(precision_interp)
    
    # Calculate macro-average for webpages
    webpage_precision_macro = np.mean(webpage_precisions, axis=0)
    webpage_recall_macro = recall_levels
    webpage_auprc = np.mean(webpage_aps) * 100
    
    # Calculate PR curves for each website class
    website_precisions = []
    website_recalls = []
    website_aps = []
    
    for i, cls in enumerate(website_classes):
        precision, recall, _ = precision_recall_curve(y_true_website_bin[:, i], y_score_website[:, i])
        ap = average_precision_score(y_true_website_bin[:, i], y_score_website[:, i])
        website_aps.append(ap)
        
        # Interpolate to common recall levels
        precision_interp = np.interp(recall_levels, recall[::-1], precision[::-1], left=0, right=1)
        website_precisions.append(precision_interp)
    
    # Calculate macro-average for websites
    website_precision_macro = np.mean(website_precisions, axis=0)
    website_recall_macro = recall_levels
    website_auprc = np.mean(website_aps) * 100
    
    return (webpage_precision_macro, webpage_recall_macro, 
            website_precision_macro, website_recall_macro, 
            None, webpage_auprc, website_auprc)


def plot_pr_curve_from_data(precision, recall, thresholds, auprc, out_file_figure, out_file_json, n_classes, level_name=""):
    """
    Plot PR curve from pre-calculated precision, recall, and thresholds.
    Used for plotting website-level PR curves calculated via mapping.
    
    Parameters:
    precision (array): Precision values at different thresholds
    recall (array): Recall values at different thresholds  
    thresholds (array or None): Threshold values (can be None for macro-average)
    auprc (float): Area under PR curve (percentage)
    out_file_figure (str): Path to save the figure
    out_file_json (str): Path to save coordinates
    n_classes (int): Number of classes
    level_name (str): Name for the level (e.g., "Webpage", "Website")
    
    Returns:
    dict: Results containing AUPRC
    """
    results = {"AUPRC": round(auprc, 2)}
    
    # Prepare data for saving
    pr_data = {
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "auprc": round(auprc, 2)
    }
    if thresholds is not None:
        pr_data["threshold"] = thresholds.tolist()
    if n_classes is not None:
        pr_data["n_classes"] = n_classes
    
    # Save coordinates to JSON
    if out_file_json:
        print(f"Saving {level_name} PR curve coordinates to: {out_file_json}")
        with open(out_file_json, 'w') as f:
            json.dump(pr_data, f, indent=4)
    
    # Plot PR curve
    if out_file_figure:
        print(f"Plotting {level_name} PR curve to: {out_file_figure}")
        plt.figure(figsize=(10, 8))
        
        try:
            # Sample points for visualization
            sample_step = max(1, len(recall) // 20)  # About 20 points
            sample_indices = np.arange(0, len(recall), sample_step)
            
            # Ensure we include the last point
            if sample_indices[-1] != len(recall) - 1:
                sample_indices = np.append(sample_indices, len(recall) - 1)
            
            recall_sampled = recall[sample_indices]
            precision_sampled = precision[sample_indices]
            
            # Apply smoothing
            if len(precision_sampled) > 5:
                window_length = min(len(precision_sampled) if len(precision_sampled) % 2 == 1 
                                  else len(precision_sampled) - 1, 11)
                if window_length >= 5:
                    polyorder = 3
                    precision_smooth = savgol_filter(precision_sampled, window_length, polyorder)
                    precision_smooth = gaussian_filter1d(precision_smooth, sigma=1.5)
                    precision_smooth = np.clip(precision_smooth, 0, 1)
                else:
                    precision_smooth = precision_sampled
            else:
                precision_smooth = precision_sampled
            
            # Plot curve and points
            plt.plot(recall_sampled, precision_smooth, color='blue', lw=2.5,
                    label=f'{level_name} (AUPRC={auprc:.2f}%)')
            plt.scatter(recall_sampled, precision_smooth, color='blue', s=50, alpha=0.7,
                       zorder=3, edgecolors='white', linewidths=1.0)
            
        except Exception as e:
            print(f"Smoothing failed: {e}, using original curve")
            plt.plot(recall, precision, color='blue', lw=2.5,
                    label=f'{level_name} (AUPRC={auprc:.2f}%)')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        title = f'{level_name} Precision-Recall Curve'
        if n_classes:
            title += f'\n({n_classes} classes)'
        plt.title(title, fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(out_file_figure, dpi=150)
        plt.close()
    
    return results
