import torch
import random
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def measurement(y_true, y_pred, eval_metrics, out_file_json=None, out_file_figure=None, reduced_unm=0):
    """
    Calculate evaluation metrics for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    eval_metrics (list): List of evaluation metrics to calculate.

    Returns:
    dict: Dictionary of calculated metrics.
    """
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
            class_accuracies = {}
            accuracies_list = []
            for cls_label in np.unique(y_true):
                mask = y_true == cls_label
                acc = round(accuracy_score(y_true[mask], y_pred[mask]) * 100, 2)
                class_accuracies[str(cls_label)] = acc
                accuracies_list.append(acc)
            if out_file_json:
                print(out_file_json)
                with open(out_file_json, 'w') as f:
                    json.dump(class_accuracies, f, indent=4)
            if out_file_figure:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(accuracies_list)), 
                    accuracies_list,
                    color='skyblue')
                plt.xlabel('Class Label')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Class-wise Accuracy Distribution')
                plt.xticks(range(len(accuracies_list)), 
                        range(len(accuracies_list)),
                        rotation=45,
                        fontsize=8)
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                plt.savefig(out_file_figure, dpi=150)
                plt.close()

        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "P@min":
            per_class_precision = precision_score(y_true, y_pred, average=None)
            results[eval_metric] = round(np.min(per_class_precision) * 100, 2)

            # Save per-class precision to JSON file
            # if out_file_json:
            #     print(out_file_json)
            #     class_precision_dict = {
            #         str(cls_label): round(prec * 100, 2)
            #         for cls_label, prec in enumerate(per_class_precision)
            #     }
            #     with open(out_file_json, 'w') as f:
            #         json.dump(class_precision_dict, f, indent=4)

            # Plot precision distribution
            # if out_file_figure:
            #     plt.figure(figsize=(12, 6))
            #     plt.bar(range(len(per_class_precision)), 
            #         [p * 100 for p in per_class_precision],
            #         color='skyblue')
            #     plt.xlabel('Class Label')
            #     plt.ylabel('Precision (%)')
            #     plt.title(f'Class-wise Precision Distribution')
            #     plt.xticks(range(len(per_class_precision)), 
            #             range(len(per_class_precision)),
            #             rotation=45,
            #             fontsize=8)
            #     plt.grid(axis='y', linestyle='--')
            #     plt.tight_layout()
            #     plt.savefig(out_file_figure, dpi=150)
            #     plt.close()

        elif eval_metric == "r-Precision":
            results[eval_metric] = round(cal_r_precision(y_true, y_pred)*100, 2)
            
            # Calculate per-class precision for closed-world classes
            open_class = y_true.max()
            closed_class_precision = precision_score(y_true, y_pred, average=None, labels=range(open_class))
            min_precision = round(np.min(closed_class_precision) * 100, 2)
                 
            # Save per-class precision to JSON file
            if out_file_json:
                print(out_file_json)
                class_precision_dict = {
                    str(cls_label): round(prec * 100, 2)
                    for cls_label, prec in enumerate(closed_class_precision)
                }
                with open(out_file_json, 'w') as f:
                    json.dump(class_precision_dict, f, indent=4)

            # Plot class-wise precision distribution
            if out_file_figure:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(closed_class_precision)), 
                    [p * 100 for p in closed_class_precision],
                    color='skyblue')
                plt.xlabel('Class Label')
                plt.ylabel('Precision (%)')
                plt.title(f'Class-wise Precision Distribution (min={min_precision}%)')
                plt.xticks(range(len(closed_class_precision)), 
                        range(len(closed_class_precision)),
                        rotation=45,
                        fontsize=8)
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                plt.savefig(out_file_figure, dpi=150)
                plt.close()
        elif eval_metric == "PR-curve":
            # PR-curve is handled separately by plot_pr_curves() function in model_utils.py
            pass
        else:
            print(eval_metric, type(eval_metric))
            raise ValueError(f"Metric {eval_metric} is not matched.")
    
    return results

def cal_r_precision(y_true, y_pred, base_r=20):
    """
    Calculate r-Precision for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    base_r (int): Base value for r-Precision calculation.

    Returns:
    float: Calculated r-Precision value.
    """
    open_class = y_true.max()  # The class index representing the open world scenario
    web2tp = {}  # True positives per class
    web2fp = {}  # False positives per class
    web2wp = {}  # Wrong predictions per class

    # Initialize dictionaries
    for web in range(open_class + 1):
        web2tp[web] = 0
        web2fp[web] = 0
        web2wp[web] = 0

    # Count true positives, false positives, and wrong predictions
    for index in range(len(y_true)):
        cur_true = y_true[index]
        cur_pred = y_pred[index]
        if cur_true == cur_pred:
            web2tp[cur_pred] += 1
        else:
            if cur_true == open_class:
                web2fp[cur_pred] += 1
            else:
                web2wp[cur_pred] += 1

    res = 0
    # Calculate r-Precision for each class
    for web in range(open_class):
        denominator = web2tp[web] + base_r * web2fp[web] + web2wp[web]
        if denominator == 0:
            continue
        res += web2tp[web] / denominator
    res /= open_class
    return res

def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad

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
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    from scipy.special import softmax
    
    # Convert logits to probabilities
    if np.max(np.abs(y_score_webpage)) > 10:
        y_score_webpage = softmax(y_score_webpage, axis=1)
    
    # Get webpage classes
    webpage_classes = np.unique(y_true_webpage)
    n_webpages = len(webpage_classes)
    
    # Handle open-world scenario: if y_score has fewer columns than classes
    if y_score_webpage.shape[1] < n_webpages:
        # Add a virtual score column for unknown class
        unknown_scores = np.max(y_score_webpage, axis=1, keepdims=True)
        y_score_webpage = np.hstack([y_score_webpage, unknown_scores])
        print(f"Added virtual score column for unknown webpage class (shape: {y_score_webpage.shape})")
    
    # Binarize webpage labels
    y_true_webpage_bin = label_binarize(y_true_webpage, classes=webpage_classes)
    if n_webpages == 2:
        y_true_webpage_bin = np.hstack([1 - y_true_webpage_bin, y_true_webpage_bin])
    
    # Map to website level
    y_true_website = mapping_array[y_true_webpage.astype(int)]
    website_classes = np.unique(y_true_website)
    n_websites = len(website_classes)
    
    # Create a mapping from website_id to index in website_classes
    website_id_to_idx = {website_id: idx for idx, website_id in enumerate(website_classes)}
    
    # Create website-level scores by aggregating webpage scores
    y_score_website = np.zeros((len(y_true_webpage), n_websites))
    for webpage_id in range(len(mapping_array)):
        if webpage_id < y_score_webpage.shape[1]:
            website_id = mapping_array[webpage_id]
            # Map website_id to its index in website_classes
            if website_id in website_id_to_idx:
                website_idx = website_id_to_idx[website_id]
                y_score_website[:, website_idx] += y_score_webpage[:, webpage_id]
    
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
    
    # For compatibility, return dummy thresholds (not used in this approach)
    thresholds = recall_levels  # Use recall levels as proxy
    
    return (webpage_precision_macro, webpage_recall_macro,
            website_precision_macro, website_recall_macro, 
            thresholds, webpage_auprc, website_auprc)

def plot_pr_curve_from_data(precision, recall, thresholds, auprc, out_file_figure=None, 
                            out_file_coords=None, n_classes=None, level_name=""):
    """
    Plot PR curve from pre-calculated precision, recall, and thresholds.
    Used for plotting website-level PR curves calculated via mapping.
    
    Parameters:
    precision (array): Precision values at different thresholds
    recall (array): Recall values at different thresholds  
    thresholds (array): Threshold values
    auprc (float): Area under PR curve (percentage)
    out_file_figure (str): Path to save the figure
    out_file_coords (str): Path to save coordinates
    n_classes (int): Number of classes
    level_name (str): Name for the level (e.g., "Webpage", "Website")
    
    Returns:
    dict: Results containing AUPRC
    """
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    
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
    if out_file_coords:
        print(f"Saving {level_name} PR curve coordinates to: {out_file_coords}")
        with open(out_file_coords, 'w') as f:
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

def plot_pr_curves(y_true, y_score, out_file_figure=None, out_file_coords=None, label_names=None):
    """
    Plot macro-average Precision-Recall curve and save coordinates.
    
    Parameters:
    y_true (array-like): True labels (shape: [n_samples])
    y_score (array-like): Prediction probabilities/logits (shape: [n_samples, n_classes])
    out_file_figure (str): Path to save the PR curve figure (PNG)
    out_file_coords (str): Path to save macro-average PR curve coordinates (JSON)
    label_names (list): Optional class names for legend
    
    Returns:
    dict: Dictionary containing macro-average AUPRC
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    from scipy.special import softmax
    
    # Convert logits to probabilities if needed
    if np.max(np.abs(y_score)) > 10:  # Likely logits, not probabilities
        y_score = softmax(y_score, axis=1)
    
    # Get unique classes
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # Handle open-world scenario: if y_score has fewer columns than classes
    # (e.g., unknown class has no corresponding score column)
    if y_score.shape[1] < n_classes:
        # Add a virtual score column for unknown class (use minimum scores)
        # For Holmes: higher similarity = worse prediction, so use max similarity (worst score)
        unknown_scores = np.max(y_score, axis=1, keepdims=True)
        y_score = np.hstack([y_score, unknown_scores])
        print(f"Added virtual score column for unknown class (shape: {y_score.shape})")
    
    # Binarize labels for multi-class PR curve calculation
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    # Calculate PR curve for each class and interpolate to common recall levels
    all_precisions = []
    all_thresholds = []
    all_aps = []
    recall_levels = np.linspace(0, 1, 101)  # 101 points from 0 to 1
    
    for i, cls in enumerate(classes):
        precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
        all_aps.append(ap)
        
        # Interpolate precision at common recall levels
        # Reverse arrays for interpolation (recall should be increasing)
        precision_interp = np.interp(recall_levels, recall[::-1], precision[::-1], left=0, right=1)
        all_precisions.append(precision_interp)
        
        # Interpolate thresholds at common recall levels
        # Note: thresholds array is one element shorter than precision/recall
        # We need to handle this properly
        if len(thresholds) > 0:
            # Extend thresholds to match recall length (last threshold corresponds to score=0)
            thresholds_extended = np.append(thresholds, 0)
            threshold_interp = np.interp(recall_levels, recall[::-1], thresholds_extended[::-1], left=1, right=0)
            all_thresholds.append(threshold_interp)
        else:
            all_thresholds.append(np.zeros_like(recall_levels))
    
    # Calculate macro-average precision and threshold at each recall level
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
    if out_file_coords:
        print(f"Saving macro-average PR curve coordinates to: {out_file_coords}")
        with open(out_file_coords, 'w') as f:
            json.dump(pr_data, f, indent=4)
    
    # Plot macro-average PR curve
    if out_file_figure:
        print(f"Plotting macro-average PR curve to: {out_file_figure}")
        plt.figure(figsize=(10, 8))
        
        # Apply enhanced smoothing for better visual quality
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter1d
        
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