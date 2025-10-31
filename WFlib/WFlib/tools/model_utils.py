import numpy as np
import torch
import os
import json
import csv
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from sklearn.metrics.pairwise import cosine_similarity
from .evaluator import measurement, plot_pr_curves, calculate_pr_curve_with_mapping, plot_pr_curve_from_data
from ..losses import HPCL


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:

            # Use get_features for DFsimCLR to extract pre-projection features
            if hasattr(net, 'get_features'):
                feature = net.get_features(data.to(device))
            else:
                feature = net(data.to(device))
                
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            
            # Use get_features for DFsimCLR to extract pre-projection features
            if hasattr(net, 'get_features'):
                feature = net.get_features(data)
            else:
                feature = net(data)
                
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())
    
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    
    return y_true, y_pred

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    return pred_labels

def fast_count_burst(arr):
    """
    Count bursts of continuous values in an array.

    Parameters:
    arr (ndarray): Input array.

    Returns:
    ndarray: Length of bursts.
    """
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs
    
    return adjusted_lengths

def model_train(
    model,
    optimizer,
    train_iter,
    valid_iter,
    loss_name,
    save_metric,
    eval_metrics,
    train_epochs,
    out_file,
    num_classes,
    num_tabs,
    device,
    lradj
):
    if loss_name in ["CrossEntropyLoss", "BCEWithLogitsLoss", "MultiLabelSoftMarginLoss"]:
        criterion = eval(f"torch.nn.{loss_name}")()
    elif loss_name == "TripletMarginLoss":
        criterion = losses.TripletMarginLoss(margin=0.1)
        miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
    elif loss_name == "SupConLoss":
        criterion = losses.SupConLoss(temperature=0.1)
    elif loss_name == "HPCL":
        criterion = HPCL(num_classes)
    elif loss_name == "MultiCrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function {loss_name} is not matched.")

    if lradj != "None":
        scheduler = eval(f"torch.optim.lr_scheduler.{lradj}")(optimizer, step_size=30, gamma=0.74)
    
    assert save_metric in eval_metrics, f"save_metric {save_metric} should be included in {eval_metrics}"
    metric_best_value = 0
    best_epoch = 0
    collected_features = None 

    for epoch in range(train_epochs):
        model.train()
        sum_loss = 0
        sum_count = 0
        
        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
            
            # Add for hierarchical labels
            # For HPCL loss, keep hierarchical structure; for others, use last column
            if len(cur_y.shape) > 1 and loss_name not in ["HPCL"]:
                cur_y = cur_y[:, -1] 

            optimizer.zero_grad()
            
            outs = model(cur_X)
            outs_for_loss = outs

            if loss_name == "TripletMarginLoss":
                hard_pairs = miner(outs, cur_y)
                loss = criterion(outs, cur_y, hard_pairs)
            elif loss_name == "SupConLoss":
                loss = criterion(outs, cur_y)
            elif loss_name == "HPCL":
                # HF model returns dict, pass entire dict to HPCL loss
                loss = criterion(outs_for_loss, cur_y)
            elif loss_name == "MultiCrossEntropyLoss":
                loss = 0
                cur_indices = torch.nonzero(cur_y)
                cur_indices = cur_indices[:,1].view(-1, num_tabs)
                for ct in range(num_tabs):
                    loss_ct = criterion(outs[:, ct], cur_indices[:, ct])
                    loss = loss + loss_ct
            else:
                loss = criterion(outs, cur_y)
            
            loss.backward()
            optimizer.step()
            
            # For loss statistics, use the correct batch size
            if loss_name == "HPCL":
                # HF model returns dict, get batch size from logits list
                actual_batch_size = outs["logits"][0].shape[0]
            else:
                actual_batch_size = outs.shape[0]
                
            sum_loss += loss.data.cpu().numpy() * actual_batch_size
            sum_count += actual_batch_size

        train_loss = round(sum_loss / sum_count, 3)
        print(f"epoch {epoch}: train_loss = {train_loss}")

        if loss_name in ["TripletMarginLoss", "SupConLoss"]:
            valid_true, valid_pred = knn_monitor(model, device, train_iter, valid_iter, num_classes, 10)
        elif loss_name == "HPCL":
            # Handle hierarchical labels for HPCL: evaluate both coarse and fine levels
            # num_classes is [num_coarse_classes, num_fine_classes_total] for HPCL
            num_coarse_classes, num_fine_classes_total = num_classes[0], num_classes[1]
            
            with torch.no_grad():
                model.eval()
                # Coarse-grained predictions and labels
                valid_pred_coarse = []
                valid_true_coarse = []
                # Fine-grained predictions and labels
                valid_pred_fine = []
                valid_true_fine = []

                for index, cur_data in enumerate(valid_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                    
                    outs = model(cur_X)
                    # Extract coarse and fine logits from list format
                    coarse_logits = outs["logits"][0]
                    fine_logits = outs["logits"][1]
                    
                    # Coarse-grained predictions
                    cur_pred_coarse = torch.argsort(coarse_logits, dim=1, descending=True)[:,0]
                    cur_y_coarse = cur_y[:, 0]  # Coarse-grained labels
                    
                    # Fine-grained predictions
                    cur_pred_fine = torch.argsort(fine_logits, dim=1, descending=True)[:,0]
                    cur_y_fine = cur_y[:, 1]  # Fine-grained labels
                    
                    valid_pred_coarse.append(cur_pred_coarse.cpu().numpy())
                    valid_true_coarse.append(cur_y_coarse.cpu().numpy())
                    valid_pred_fine.append(cur_pred_fine.cpu().numpy())
                    valid_true_fine.append(cur_y_fine.cpu().numpy())
                
                valid_pred_coarse = np.concatenate(valid_pred_coarse)
                valid_true_coarse = np.concatenate(valid_true_coarse)
                valid_pred_fine = np.concatenate(valid_pred_fine)
                valid_true_fine = np.concatenate(valid_true_fine)
                
                # Use fine-grained results for save_metric
                valid_pred = valid_pred_fine
                valid_true = valid_true_fine
        else:
            with torch.no_grad():
                model.eval()
                sum_loss = 0
                sum_count = 0
                valid_pred = []
                valid_true = []

                for index, cur_data in enumerate(valid_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)

                    # Add for hierarchical labels
                    # For HPCL loss, keep hierarchical structure; for others, use last column
                    if len(cur_y.shape) > 1 and cur_y.shape[1] > 1 and loss_name != "HPCL":
                        cur_y = cur_y[:, -1] 
                    
                    outs = model(cur_X)
                    
                    
                    # Extract logits from HF model dict output for HPCL
                    if loss_name == "HPCL":
                        # Use fine_logits for standard validation (webpage-level) - list format
                        fine_logits = outs["logits"][1]
                        cur_pred = torch.argsort(fine_logits, dim=1, descending=True)[:,0]
                    elif loss_name in ["BCEWithLogitsLoss", "MultiLabelSoftMarginLoss"]:
                        cur_pred = torch.sigmoid(outs)
                    elif loss_name == "CrossEntropyLoss":
                        cur_pred = torch.argsort(outs, dim=1, descending=True)[:,0]
                    elif loss_name == "MultiCrossEntropyLoss":
                        cur_indices = torch.argmax(outs, dim=-1).cpu()
                        cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                        for cur_tab in range(cur_indices.shape[1]):
                            row_indices = torch.arange(cur_pred.shape[0])
                            cur_pred[row_indices,cur_indices[:,cur_tab]] += 1
                    else:
                        raise ValueError(f"Loss function {loss_name} is not matched.")

                    valid_pred.append(cur_pred.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())
                
                valid_pred = np.concatenate(valid_pred)
                valid_true = np.concatenate(valid_true)
        
        valid_result = measurement(valid_true, valid_pred, eval_metrics)
        
        if loss_name == "HPCL":
            # Also calculate and display coarse-level metrics
            valid_result_coarse = measurement(valid_true_coarse, valid_pred_coarse, eval_metrics)
            print(f"{epoch}: (Webpage) {valid_result}")
            print(f"{epoch}: (Website) {valid_result_coarse}")
        else:
            print(f"{epoch}: {valid_result}")
        
        if valid_result[save_metric] > metric_best_value:
            metric_best_value = valid_result[save_metric]
            best_epoch = epoch
            torch.save(model.state_dict(), out_file)
            
        
        print(f"best epoch {best_epoch}: {save_metric}={metric_best_value}")
        
        if lradj != "None":
            scheduler.step()

def model_eval(
        model, 
        test_iter, 
        valid_iter, 
        eval_method, 
        eval_metrics, 
        out_file, 
        num_classes, 
        ckp_path, 
        scenario,
        num_tabs,
        device,
        mapping_file
    ):
    
    # # ============================================================
    # # Model Performance Analysis
    # # ============================================================
    # import time
    
    # perf_metrics = {}
    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # perf_metrics['params_M'] = total_params / 1e6
    # perf_metrics['trainable_params_M'] = trainable_params / 1e6
    
    # try:
    #     from thop import profile
    #     model.eval()
        
    #     for sample_data in test_iter:
    #         sample_input = sample_data[0][0:1].to(device)
    #         break
        
    #     if isinstance(num_classes, list) and len(num_classes) == 2:
    #         class ModelWrapper(torch.nn.Module):
    #             def __init__(self, model):
    #                 super().__init__()
    #                 self.model = model
    #             def forward(self, x):
    #                 output = self.model(x)
    #                 return output["logits"][0] if isinstance(output, dict) else output
    #         wrapped_model = ModelWrapper(model)
    #         flops, _ = profile(wrapped_model, inputs=(sample_input,), verbose=False)
    #     else:
    #         flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        
    #     perf_metrics['flops_G'] = flops / 1e9
    # except ImportError:
    #     perf_metrics['flops_G'] = None
    #     perf_metrics['flops_note'] = "thop not installed"
    # except Exception as e:
    #     perf_metrics['flops_G'] = None
    #     perf_metrics['flops_note'] = "calculation failed"
    
    # model.eval()
    # warmup_runs = 10
    # test_runs = 100
    
    # for sample_data in test_iter:
    #     sample_input = sample_data[0][0:1].to(device)
    #     sample_label = sample_data[1][0:1]
    #     break
    
    # if eval_method == "Holmes":
    #     spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
    #     if os.path.exists(spatial_dist_file):
    #         spatial_data = np.load(spatial_dist_file)
            
    #         is_hierarchical_holmes = isinstance(num_classes, (list, tuple)) and len(num_classes) == 2
            
    #         if is_hierarchical_holmes:
    #             centroids = spatial_data["centroid"]
    #             radii = spatial_data["radius"]
    #             num_websites = num_classes[0]
    #             num_webpages = num_classes[1]
    #             website_centroids = centroids[0]
    #             webpage_centroids = centroids[1]
    #             website_radii = radii[0]
    #             webpage_radii = radii[1]
    #         else:
    #             webs_centroid = spatial_data["centroid"]
    #             webs_radius = spatial_data["radius"]
            
    #         with torch.no_grad():
    #             for _ in range(warmup_runs):
    #                 if hasattr(model, 'get_features'):
    #                     embs = model.get_features(sample_input).cpu().numpy()
    #                 else:
    #                     embs = model(sample_input).cpu().numpy()
                    
    #                 if is_hierarchical_holmes:
    #                     website_sims = 1 - cosine_similarity(embs, website_centroids[:num_websites])
    #                     website_sims -= website_radii[:num_websites]
    #                     _ = np.argmin(website_sims, axis=1)
                        
    #                     webpage_sims = 1 - cosine_similarity(embs, webpage_centroids[:num_webpages])
    #                     webpage_sims -= webpage_radii[:num_webpages]
    #                     _ = np.argmin(webpage_sims, axis=1)
    #                 else:
    #                     all_sims = 1 - cosine_similarity(embs, webs_centroid)
    #                     all_sims -= webs_radius
    #                     _ = np.argmin(all_sims, axis=1)
                    
    #                 if device.type == 'cuda':
    #                     torch.cuda.synchronize()
            
    #         inference_times = []
    #         with torch.no_grad():
    #             for _ in range(test_runs):
    #                 if device.type == 'cuda':
    #                     torch.cuda.synchronize()
                    
    #                 start = time.time()
                    
    #                 if hasattr(model, 'get_features'):
    #                     embs = model.get_features(sample_input).cpu().numpy()
    #                 else:
    #                     embs = model(sample_input).cpu().numpy()
                    
    #                 if is_hierarchical_holmes:
    #                     website_sims = 1 - cosine_similarity(embs, website_centroids[:num_websites])
    #                     website_sims -= website_radii[:num_websites]
    #                     _ = np.argmin(website_sims, axis=1)
                        
    #                     webpage_sims = 1 - cosine_similarity(embs, webpage_centroids[:num_webpages])
    #                     webpage_sims -= webpage_radii[:num_webpages]
    #                     _ = np.argmin(webpage_sims, axis=1)
    #                 else:
    #                     all_sims = 1 - cosine_similarity(embs, webs_centroid)
    #                     all_sims -= webs_radius
    #                     _ = np.argmin(all_sims, axis=1)
                    
    #                 end = time.time()
    #                 inference_times.append((end - start) * 1000)
            
    #         perf_metrics['inference_time_ms'] = np.mean(inference_times)
    #         perf_metrics['inference_time_std_ms'] = np.std(inference_times)
    #         perf_metrics['inference_note'] = "full pipeline (w/ clustering)"
    #     else:
    #         perf_metrics['inference_time_ms'] = None
    #         perf_metrics['inference_note'] = "spatial_distribution.npz not found"
    
    # else:
    #     with torch.no_grad():
    #         for _ in range(warmup_runs):
    #             _ = model(sample_input)
    #             if device.type == 'cuda':
    #                 torch.cuda.synchronize()
        
    #     inference_times = []
    #     with torch.no_grad():
    #         for _ in range(test_runs):
    #             if device.type == 'cuda':
    #                 torch.cuda.synchronize()
    #                 start = time.time()
    #                 _ = model(sample_input)
    #                 torch.cuda.synchronize()
    #                 end = time.time()
    #             else:
    #                 start = time.time()
    #                 _ = model(sample_input)
    #                 end = time.time()
    #             inference_times.append((end - start) * 1000)
        
    #     perf_metrics['inference_time_ms'] = np.mean(inference_times)
    #     perf_metrics['inference_time_std_ms'] = np.std(inference_times)
    #     perf_metrics['inference_note'] = "model forward only"
    
    # print("\nModel Performance Analysis:")
    # print(f"  - FLOPs: per-sample forward pass")
    # print(f"  - Inference: inference time ({perf_metrics.get('inference_note', '')})")
    
    # print("\n" + "="*70)
    # print(f"{'Model':<15} {'#Params (M)':<15} {'FLOPs (G)':<15} {'Inference (ms)':<20}")
    # print("-"*70)
    
    # model_name = os.path.basename(ckp_path) if ckp_path else "Unknown"
    
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

    # Initializing hierarchical label related variables
    is_hierarchical = False
    website_y_pred = None
    website_y_true = None
    
    # Check if this is HF with hierarchical labels
    if isinstance(num_classes, list) and len(num_classes) == 2:
        is_hierarchical = True
        num_coarse_classes, num_fine_classes_total = num_classes[0], num_classes[1]
        print(f"Detected hierarchical testing: {num_coarse_classes} coarse classes, {num_fine_classes_total} total fine classes")
    
    if eval_method == "common":
        with torch.no_grad():
            model.eval()
            # # Start collecting features
            # model.start_collecting()
            y_pred = []
            y_true = []
            all_logits = []  # Save all logits for Top-k calculation
            
            # For HPCL hierarchical testing
            if is_hierarchical:
                y_pred_coarse = []
                y_true_coarse = []
                y_pred_fine = []
                y_true_fine = []
                all_coarse_logits = []  # Save coarse logits for PR curve

            for index, cur_data in enumerate(test_iter):
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                outs = model(cur_X)
                
                # Handle HF model dict output with list format
                if isinstance(outs, dict) and "logits" in outs and isinstance(outs["logits"], list):
                    # New architecture: logits in list format [coarse_logits, fine_logits]
                    coarse_logits = outs["logits"][0]
                    fine_logits = outs["logits"][1]
                    outs_for_pred = fine_logits  # Use fine_logits for non-hierarchical case
                else:
                    outs_for_pred = outs
                
                if is_hierarchical:
                    # Handle HPCL hierarchical predictions using list format
                    if isinstance(outs, dict) and "logits" in outs and isinstance(outs["logits"], list):
                        # Already extracted above
                        pass
                    else:
                        # Fallback to slicing for backward compatibility
                        coarse_logits = outs_for_pred[:, :num_coarse_classes]
                        fine_logits = outs_for_pred[:, num_coarse_classes:]
                    
                    # Save both coarse and fine logits for PR curves
                    all_coarse_logits.append(coarse_logits.cpu().numpy())
                    all_logits.append(fine_logits.cpu().numpy())
                    
                    cur_pred_coarse = torch.argsort(coarse_logits, dim=1, descending=True)[:,0]
                    cur_pred_fine = torch.argsort(fine_logits, dim=1, descending=True)[:,0]
                    
                    cur_y_coarse = cur_y[:, 0]  # Coarse-grained labels
                    cur_y_fine = cur_y[:, 1]    # Fine-grained labels
                    
                    y_pred_coarse.append(cur_pred_coarse.cpu().numpy())
                    y_true_coarse.append(cur_y_coarse.cpu().numpy())
                    y_pred_fine.append(cur_pred_fine.cpu().numpy())
                    y_true_fine.append(cur_y_fine.cpu().numpy())
                    
                    # Use fine-grained for main results
                    cur_pred = cur_pred_fine
                    cur_y = cur_y_fine
                else:
                    # For non-hierarchical models, save complete logits
                    all_logits.append(outs_for_pred.cpu().numpy())
                    
                    if num_tabs == 1:
                        cur_pred = torch.argsort(outs_for_pred, dim=1, descending=True)[:,0]
                    else:
                        if len(outs_for_pred.shape) <= 2:
                            cur_pred = torch.sigmoid(outs_for_pred)
                        else:
                            cur_indices = torch.argmax(outs_for_pred, dim=-1).cpu()
                            cur_pred = torch.zeros((cur_indices.shape[0], num_classes))
                            for cur_tab in range(cur_indices.shape[1]):
                                row_indices = torch.arange(cur_pred.shape[0])
                                cur_pred[row_indices,cur_indices[:,cur_tab]] += 1

                y_pred.append(cur_pred.cpu().numpy())
                y_true.append(cur_y.cpu().numpy())

            # # Stop collecting features
            # model.stop_collecting()
            
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            all_logits = np.concatenate(all_logits, axis=0)
            
            # Handle HPCL hierarchical results
            if is_hierarchical:
                y_pred_coarse = np.concatenate(y_pred_coarse)
                y_true_coarse = np.concatenate(y_true_coarse)
                y_pred_fine = np.concatenate(y_pred_fine)
                y_true_fine = np.concatenate(y_true_fine)
                all_coarse_logits = np.concatenate(all_coarse_logits, axis=0)
            
            # # Save features
            # if hasattr(model, 'collected_features') and model.collected_features is not None:
            #     collected_features = model.collected_features.numpy()
            #     feature_file = os.path.splitext(out_file)[0] + '_features.npz'
            #     np.savez(feature_file, 
            #             features=collected_features,
            #             preds=y_pred,
            #             labels=y_true)
                
    elif eval_method == "kNN":
        y_true, y_pred = knn_monitor(model, device, valid_iter, test_iter, num_classes, 10)
    elif eval_method == "Holmes":
        open_threshold = 1e-2
        spatial_dist_file = os.path.join(ckp_path, "spatial_distribution.npz")
        assert os.path.exists(spatial_dist_file), f"{spatial_dist_file} does not exist, please run spatial_analysis.py first"
        spatial_data = np.load(spatial_dist_file)
        
        # Handle hierarchical labels: evaluate both website and webpage levels
        is_hierarchical = isinstance(num_classes, (list, tuple)) and len(num_classes) == 2
        
        if is_hierarchical:
            centroids = spatial_data["centroid"]  # shape: (2, max_classes, feature_dim)
            radii = spatial_data["radius"]        # shape: (2, max_classes) 
            num_websites = num_classes[0]
            num_webpages = num_classes[1]
            
            website_centroids = centroids[0]  # (max_classes, feature_dim)
            webpage_centroids = centroids[1]  # (max_classes, feature_dim)
            website_radii = radii[0]          # (max_classes,)
            webpage_radii = radii[1]          # (max_classes,)
            
            with torch.no_grad():
                model.eval()

                website_pred = []
                website_true = []
                webpage_pred = []
                webpage_true = []
                all_similarities = []  # Save webpage similarities for top-k and PR curve
                all_website_similarities = []  # Save website similarities for PR curve

                for index, cur_data in enumerate(test_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                    # Use get_features for DFsimCLR to extract pre-projection features (consistent with knn_monitor)
                    if hasattr(model, 'get_features'):
                        embs = model.get_features(cur_X).cpu().numpy()
                    else:
                        embs = model(cur_X).cpu().numpy()
                    cur_y = cur_y.cpu().numpy()


                    website_sims = 1 - cosine_similarity(embs, website_centroids[:num_websites])
                    website_sims -= website_radii[:num_websites]
                    website_preds = np.argmin(website_sims, axis=1)
                    
                    webpage_sims = 1 - cosine_similarity(embs, webpage_centroids[:num_webpages])
                    webpage_sims -= webpage_radii[:num_webpages]
                    webpage_preds = np.argmin(webpage_sims, axis=1)
                    

                    if scenario == "Open-world":
                        website_dists = np.min(website_sims, axis=1)
                        website_open_indices = np.where(website_dists > open_threshold)[0]
                        website_preds[website_open_indices] = num_websites - 1
                        
                        webpage_dists = np.min(webpage_sims, axis=1)
                        webpage_open_indices = np.where(webpage_dists > open_threshold)[0]
                        webpage_preds[webpage_open_indices] = num_webpages - 1
                    

                    website_pred.append(website_preds)
                    website_true.append(cur_y[:, 0])
                    webpage_pred.append(webpage_preds)
                    webpage_true.append(cur_y[:, 1])
                    all_similarities.append(webpage_sims)  # Save webpage similarities for analysis
                    all_website_similarities.append(website_sims)  # Save website similarities for PR curve
                    

                y_pred = np.concatenate(webpage_pred).flatten()
                y_true = np.concatenate(webpage_true).flatten()
                all_similarities = np.concatenate(all_similarities, axis=0)
                all_website_similarities = np.concatenate(all_website_similarities, axis=0)
                
                website_y_pred = np.concatenate(website_pred).flatten()
                website_y_true = np.concatenate(website_true).flatten()
        else:
            webs_centroid = spatial_data["centroid"]
            webs_radius = spatial_data["radius"]

            with torch.no_grad():
                model.eval()
                y_pred = []
                y_true = []
                all_similarities = []  # Save all similarities for top-k calculation

                for index, cur_data in enumerate(test_iter):
                    cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
                    embs = model(cur_X).cpu().numpy()
                    cur_y = cur_y.cpu().numpy()

                    all_sims = 1 - cosine_similarity(embs, webs_centroid)
                    all_sims -= webs_radius
                    outs = np.argmin(all_sims, axis=1)

                    if scenario == "Open-world":
                        outs_d = np.min(all_sims, axis=1)
                        open_indices = np.where(outs_d > open_threshold)[0]
                        outs[open_indices] = num_classes - 1

                    y_pred.append(outs)
                    y_true.append(cur_y)
                    all_similarities.append(all_sims)
                    
                y_pred = np.concatenate(y_pred).flatten()
                y_true = np.concatenate(y_true).flatten()
                all_similarities = np.concatenate(all_similarities, axis=0)
    else:
        raise ValueError(f"Evaluation method {eval_method} is not matched.")
        
    # Record mismatched predictions to CSV file
    mismatch_csv_path = os.path.join(ckp_path, 'prediction_mismatches.csv')
    mismatches = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            mismatches.append([i, y_true[i], y_pred[i]])
    
    if mismatches:
        with open(mismatch_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample_Index', 'True_Label', 'Predicted_Label'])
            writer.writerows(mismatches)
    
    # Calculate Top-k accuracy for common method
    if eval_method == "common" and 'all_logits' in locals():
        print(f"\nCalculating Top-K accuracy for all samples:")
        
        # Process prediction result format
        if len(y_pred.shape) == 1:  # Single-label classification
            true_labels = y_true.flatten()
            pred_logits = all_logits
        else:  # Multi-label or other formats
            true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            pred_logits = all_logits
        
        # Calculate top-k accuracy for different k values
        k_values = [1, 2, 3, 4, 5]
        topk_results = {}
        
        # Get actual number of classes
        actual_num_classes = pred_logits.shape[1] if len(pred_logits.shape) > 1 else num_classes
        
        for k in k_values:
            if k > actual_num_classes:
                continue
                
            correct_count = 0
            for i, true_label in enumerate(true_labels):
                # Get top-k classes with highest logits (highest probabilities)
                topk_indices = np.argpartition(pred_logits[i], -k)[-k:]
                
                # Check if true label is in top-k
                if true_label in topk_indices:
                    correct_count += 1
            
            topk_accuracy = correct_count / len(true_labels)
            topk_results[f'top_{k}'] = topk_accuracy
            print(f"  Top-{k} accuracy: {topk_accuracy:.4f} ({correct_count}/{len(true_labels)})")
        
        # Save top-k results to JSON file
        topk_results_path = os.path.join(ckp_path, 'topk_accuracy_common.json')
        with open(topk_results_path, 'w') as f:
            json.dump(topk_results, f, indent=4)
    
    # Calculate overall Top-k accuracy for Holmes method
    if eval_method == "Holmes" and 'all_similarities' in locals():
        
        # Calculate top-k accuracy for different k values
        k_values = [1, 2, 3, 4, 5]
        topk_results = {}
        
        # Get actual number of classes (handle hierarchical case)
        actual_num_classes = num_classes[1] if isinstance(num_classes, (list, tuple)) else num_classes
        
        total_samples = len(y_true)
        
        for k in k_values:
            if k > actual_num_classes:
                continue
                
            correct_count = 0
            for i, true_label in enumerate(y_true):
                # Get top-k classes with smallest similarity (nearest distances)
                topk_indices = np.argpartition(all_similarities[i], k-1)[:k]
                
                # Check if true label is in top-k
                if true_label in topk_indices:
                    correct_count += 1
            
            topk_accuracy = correct_count / total_samples
            topk_results[f'top_{k}'] = topk_accuracy
            print(f"  Top-{k} accuracy: {topk_accuracy:.4f} ({correct_count}/{total_samples})")
        
        # Save top-k results to JSON file
        topk_results_path = os.path.join(ckp_path, 'topk_accuracy_holmes.json')
        with open(topk_results_path, 'w') as f:
            json.dump(topk_results, f, indent=4)

    if eval_method == "Holmes" and is_hierarchical:
        webpage_acc_json = os.path.join(ckp_path, 'acc_webpage.json')
        webpage_acc_fig = os.path.join(ckp_path, 'acc_webpage.png')
        website_acc_json = os.path.join(ckp_path, 'acc_website.json')
        website_acc_fig = os.path.join(ckp_path, 'acc_website.png')
        
        webpage_result = measurement(y_true, y_pred, eval_metrics, webpage_acc_json, webpage_acc_fig)
        website_result = measurement(website_y_true, website_y_pred, eval_metrics, website_acc_json, website_acc_fig)      
        print(f"(Webpage) {webpage_result}")
        print(f"(Website) {website_result}")
        
        combined_result = {
            "webpage": webpage_result,
            "website": website_result
        }
        
        # Open-world scenario: additionally calculate metrics excluding unknown class
        if scenario == "Open-world":
            num_webpages = num_classes[1]
            unknown_class_webpage = num_webpages - 1
            
            # Filter out samples where predicted or true label is unknown class
            known_indices = np.where((y_true != unknown_class_webpage) & (y_pred != unknown_class_webpage))[0]
            
            if len(known_indices) > 0:
                y_true_known = y_true[known_indices]
                y_pred_known = y_pred[known_indices]
                website_y_true_known = website_y_true[known_indices]
                website_y_pred_known = website_y_pred[known_indices]
                
                webpage_result_known = measurement(y_true_known, y_pred_known, eval_metrics)
                website_result_known = measurement(website_y_true_known, website_y_pred_known, eval_metrics)
                
                print(f"(Webpage - Known Only) {webpage_result_known}")
                print(f"(Website - Known Only) {website_result_known}")
                
                combined_result["webpage_known_only"] = webpage_result_known
                combined_result["website_known_only"] = website_result_known
                combined_result["known_samples_count"] = int(len(known_indices))
                combined_result["total_samples_count"] = int(len(y_true))
        
        # Plot PR curves if requested and similarities are available
        # (placed after known_only calculations for proper output order)
        # Note: For Holmes, we use negative similarities as logits (lower similarity = higher confidence)
        if 'PR-curve' in eval_metrics and 'all_similarities' in locals() and all_similarities is not None:
            # PR curve for webpage level (Holmes hierarchical)
            pr_curve_fig_webpage = os.path.join(ckp_path, 'pr_curve_webpage.png')
            pr_curve_coords_webpage = os.path.join(ckp_path, 'pr_curve_coords_webpage.json')
            pr_results_webpage = plot_pr_curves(y_true, -all_similarities, pr_curve_fig_webpage, pr_curve_coords_webpage)
            print(f"PR-Curve Results (Webpage): {pr_results_webpage}")
            
            # PR curve for website level (Holmes hierarchical)
            pr_curve_fig_website = os.path.join(ckp_path, 'pr_curve_website.png')
            pr_curve_coords_website = os.path.join(ckp_path, 'pr_curve_coords_website.json')
            pr_results_website = plot_pr_curves(website_y_true, -all_website_similarities, pr_curve_fig_website, pr_curve_coords_website)
            print(f"PR-Curve Results (Website): {pr_results_website}")
       
        with open(out_file, "w") as fp:
            json.dump(combined_result, fp, indent=4)
    else:
        if mapping_file is not None:
            mapping_array = np.load(mapping_file)
            print(f"Detected mapping file")
 
            website_y_true = mapping_array[y_true.astype(int)]
            website_y_pred = mapping_array[y_pred.astype(int)]
             
            webpage_acc_json = os.path.join(ckp_path, 'acc_webpage.json')
            webpage_acc_fig = os.path.join(ckp_path, 'acc_webpage.png')
            website_acc_json = os.path.join(ckp_path, 'acc_website.json')
            website_acc_fig = os.path.join(ckp_path, 'acc_website.png')
            
            webpage_result = measurement(y_true, y_pred, eval_metrics, webpage_acc_json, webpage_acc_fig)
            website_result = measurement(website_y_true, website_y_pred, eval_metrics, website_acc_json, website_acc_fig)      
            print(f"(Webpage) {webpage_result}")
            print(f"(Website) {website_result}")
                      
            combined_result = {
                "webpage": webpage_result,
                "website": website_result
            }
            
            # Open-world scenario: additionally calculate metrics excluding unknown class
            if scenario == "Open-world":
                # Get unknown class label (last class)
                unknown_class_webpage = int(np.max(y_true))
                
                # Filter out samples where predicted or true label is unknown class
                known_indices = np.where((y_true != unknown_class_webpage) & (y_pred != unknown_class_webpage))[0]
                
                if len(known_indices) > 0:
                    y_true_known = y_true[known_indices]
                    y_pred_known = y_pred[known_indices]
                    website_y_true_known = website_y_true[known_indices]
                    website_y_pred_known = website_y_pred[known_indices]
                    
                    webpage_result_known = measurement(y_true_known, y_pred_known, eval_metrics)
                    website_result_known = measurement(website_y_true_known, website_y_pred_known, eval_metrics)
                    
                    print(f"(Webpage - Known Only) {webpage_result_known}")
                    print(f"(Website - Known Only) {website_result_known}")
                    
                    combined_result["webpage_known_only"] = webpage_result_known
                    combined_result["website_known_only"] = website_result_known
                    combined_result["known_samples_count"] = int(len(known_indices))
                    combined_result["total_samples_count"] = int(len(y_true))
            
            # Plot PR curves if requested
            # (placed after known_only calculations for proper output order)
            if 'PR-curve' in eval_metrics:
                # Check if we have logits (for common method) or similarities (for Holmes method)
                if 'all_logits' in locals() and all_logits is not None:
                    # Common method with logits
                    pr_curve_fig_webpage = os.path.join(ckp_path, 'pr_curve_webpage.png')
                    pr_curve_coords_webpage = os.path.join(ckp_path, 'pr_curve_coords_webpage.json')
                    pr_results_webpage = plot_pr_curves(y_true, all_logits, pr_curve_fig_webpage, pr_curve_coords_webpage)
                    print(f"PR-Curve Results (Webpage): {pr_results_webpage}")
                    
                    # Calculate and plot website PR curve using mapping (no direct logits)
                    _, _, website_prec, website_rec, _, _, website_auprc = \
                        calculate_pr_curve_with_mapping(y_true, all_logits, mapping_array)
                    
                    pr_curve_fig_website = os.path.join(ckp_path, 'pr_curve_website.png')
                    pr_curve_coords_website = os.path.join(ckp_path, 'pr_curve_coords_website.json')
                    pr_results_website = plot_pr_curve_from_data(
                        website_prec, website_rec, None, website_auprc,
                        pr_curve_fig_website, pr_curve_coords_website,
                        n_classes=len(np.unique(website_y_true)), level_name="Website"
                    )
                    print(f"PR-Curve Results (Website): {pr_results_website}")
                    
                elif 'all_similarities' in locals() and all_similarities is not None:
                    # Holmes method with similarities (use negative similarities as logits)
                    pr_curve_fig_webpage = os.path.join(ckp_path, 'pr_curve_webpage.png')
                    pr_curve_coords_webpage = os.path.join(ckp_path, 'pr_curve_coords_webpage.json')
                    pr_results_webpage = plot_pr_curves(y_true, -all_similarities, pr_curve_fig_webpage, pr_curve_coords_webpage)
                    print(f"PR-Curve Results (Webpage): {pr_results_webpage}")
                    
                    # Calculate and plot website PR curve using mapping
                    _, _, website_prec, website_rec, _, _, website_auprc = \
                        calculate_pr_curve_with_mapping(y_true, -all_similarities, mapping_array)
                    
                    pr_curve_fig_website = os.path.join(ckp_path, 'pr_curve_website.png')
                    pr_curve_coords_website = os.path.join(ckp_path, 'pr_curve_coords_website.json')
                    pr_results_website = plot_pr_curve_from_data(
                        website_prec, website_rec, None, website_auprc,
                        pr_curve_fig_website, pr_curve_coords_website,
                        n_classes=len(np.unique(website_y_true)), level_name="Website"
                    )
                    print(f"PR-Curve Results (Website): {pr_results_website}")
           
            with open(out_file, "w") as fp:
                json.dump(combined_result, fp, indent=4)
        else:
            if is_hierarchical:
                # Handle HPCL hierarchical results
                coarse_acc_json = os.path.join(ckp_path, 'acc_website.json')
                coarse_acc_fig = os.path.join(ckp_path, 'acc_website.png')
                fine_acc_json = os.path.join(ckp_path, 'acc_webpage.json')
                fine_acc_fig = os.path.join(ckp_path, 'acc_webpage.png')
                
                website_result = measurement(y_true_coarse, y_pred_coarse, eval_metrics, coarse_acc_json, coarse_acc_fig)
                webpage_result = measurement(y_true_fine, y_pred_fine, eval_metrics, fine_acc_json, fine_acc_fig)
                print(f"(Webpage) {webpage_result}")
                print(f"(Website) {website_result}")

                
                combined_result = {
                    "Webpage": webpage_result,
                    "Website": website_result

                }
                
                # Open-world scenario: additionally calculate metrics excluding unknown class
                if scenario == "Open-world":
                    # Get unknown class label (last class)
                    unknown_class_fine = int(np.max(y_true_fine))
                    unknown_class_coarse = int(np.max(y_true_coarse))
                    
                    # Filter out samples where predicted or true label is unknown class
                    known_indices_fine = np.where((y_true_fine != unknown_class_fine) & (y_pred_fine != unknown_class_fine))[0]
                    known_indices_coarse = np.where((y_true_coarse != unknown_class_coarse) & (y_pred_coarse != unknown_class_coarse))[0]
                    
                    if len(known_indices_fine) > 0:
                        y_true_fine_known = y_true_fine[known_indices_fine]
                        y_pred_fine_known = y_pred_fine[known_indices_fine]
                        
                        webpage_result_known = measurement(y_true_fine_known, y_pred_fine_known, eval_metrics)
                        print(f"(Webpage - Known Only) {webpage_result_known}")
                        
                        combined_result["Webpage_known_only"] = webpage_result_known
                        combined_result["known_samples_count_fine"] = int(len(known_indices_fine))
                    
                    if len(known_indices_coarse) > 0:
                        y_true_coarse_known = y_true_coarse[known_indices_coarse]
                        y_pred_coarse_known = y_pred_coarse[known_indices_coarse]
                        
                        website_result_known = measurement(y_true_coarse_known, y_pred_coarse_known, eval_metrics)
                        print(f"(Website - Known Only) {website_result_known}")
                        
                        combined_result["Website_known_only"] = website_result_known
                        combined_result["known_samples_count_coarse"] = int(len(known_indices_coarse))
                    
                    combined_result["total_samples_count"] = int(len(y_true_fine))
                
                # Plot PR curves for hierarchical model if requested and logits are available
                # (placed after known_only calculations for proper output order)
                if 'PR-curve' in eval_metrics and 'all_logits' in locals() and all_logits is not None:
                    # Plot webpage (fine-grained) PR curve
                    pr_curve_fig_fine = os.path.join(ckp_path, 'pr_curve_webpage.png')
                    pr_curve_coords_fine = os.path.join(ckp_path, 'pr_curve_coords_webpage.json')
                    pr_results_fine = plot_pr_curves(y_true_fine, all_logits, pr_curve_fig_fine, pr_curve_coords_fine)
                    print(f"PR-Curve Results (Webpage): {pr_results_fine}")
                    
                    # Plot website (coarse-grained) PR curve
                    pr_curve_fig_coarse = os.path.join(ckp_path, 'pr_curve_website.png')
                    pr_curve_coords_coarse = os.path.join(ckp_path, 'pr_curve_coords_website.json')
                    pr_results_coarse = plot_pr_curves(y_true_coarse, all_coarse_logits, pr_curve_fig_coarse, pr_curve_coords_coarse)
                    print(f"PR-Curve Results (Website): {pr_results_coarse}")
                
                with open(out_file, "w") as fp:
                    json.dump(combined_result, fp, indent=4)
            else:
                acc_json = os.path.join(ckp_path, 'acc.json')
                acc_fig = os.path.join(ckp_path, 'acc.png')
                result = measurement(y_true, y_pred, eval_metrics, acc_json, acc_fig)
                print(result)
                
                # Open-world scenario: additionally calculate metrics excluding unknown class
                if scenario == "Open-world":
                    # Get unknown class label (last class)
                    unknown_class = int(np.max(y_true))
                    
                    # Filter out samples where predicted or true label is unknown class
                    known_indices = np.where((y_true != unknown_class) & (y_pred != unknown_class))[0]
                    
                    if len(known_indices) > 0:
                        y_true_known = y_true[known_indices]
                        y_pred_known = y_pred[known_indices]
                        
                        result_known = measurement(y_true_known, y_pred_known, eval_metrics)
                        print(f"(Known Only) {result_known}")
                        
                        # Plot PR curves if requested and logits are available
                        # (placed after known_only calculations for proper output order)
                        if 'PR-curve' in eval_metrics and 'all_logits' in locals() and all_logits is not None:
                            pr_curve_fig = os.path.join(ckp_path, 'pr_curve.png')
                            pr_curve_coords = os.path.join(ckp_path, 'pr_curve_coords.json')
                            pr_results = plot_pr_curves(y_true, all_logits, pr_curve_fig, pr_curve_coords)
                            print(f"PR-Curve Results: {pr_results}")
                        
                        # Combine results
                        combined_result = {
                            "all_samples": result,
                            "known_only": result_known,
                            "known_samples_count": int(len(known_indices)),
                            "total_samples_count": int(len(y_true))
                        }
                        
                        with open(out_file, "w") as fp:
                            json.dump(combined_result, fp, indent=4)
                    else:
                        with open(out_file, "w") as fp:
                            json.dump(result, fp, indent=4)
                else:
                    # Plot PR curves if no open-world scenario
                    if 'PR-curve' in eval_metrics and 'all_logits' in locals() and all_logits is not None:
                        pr_curve_fig = os.path.join(ckp_path, 'pr_curve.png')
                        pr_curve_coords = os.path.join(ckp_path, 'pr_curve_coords.json')
                        pr_results = plot_pr_curves(y_true, all_logits, pr_curve_fig, pr_curve_coords)
                        print(f"PR-Curve Results: {pr_results}")
                    
                    with open(out_file, "w") as fp:
                        json.dump(result, fp, indent=4)

def info_nce_loss(features, batch_size, device):
    """
    Compute the InfoNCE loss.

    Parameters:
    features (Tensor): Feature tensor.
    batch_size (int): Batch size.
    device (torch.device): The device to run the computations on.

    Returns:
    tuple: Logits and labels.
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / 0.5

    return logits, labels

def pretrian_accuracy(output, target):
    """
    Compute the accuracy over the top predictions.

    Parameters:
    output (Tensor): Model output.
    target (Tensor): Target labels.

    Returns:
    float: Computed accuracy.
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)

        return res.cpu().numpy()[0]

def model_pretrian(model, optimizer, train_iter, train_epochs, out_file, batch_size, device):
    """
    Pretrain the model.

    Parameters:
    model (nn.Module): The neural network model.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    train_iter (DataLoader): DataLoader for training data.
    train_epochs (int): Number of training epochs.
    out_file (str): Output file to save the model.
    batch_size (int): Batch size.
    device (torch.device): The device to run the computations on.
    """
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(train_epochs):
        model.train()
        mean_acc = 0
        iter_count = 0

        for index, cur_data in enumerate(train_iter):
            cur_X, cur_y = cur_data[0], cur_data[1]
            cur_X = torch.cat(cur_X, dim=0)
            cur_X = cur_X.view(cur_X.size(0), 1, cur_X.size(1)).float().to(device)

            optimizer.zero_grad()
            features = model(cur_X)
            logits, labels = info_nce_loss(features, batch_size, device)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            iter_count += 1
            mean_acc += pretrian_accuracy(logits, labels)

        mean_acc /= iter_count
        print(f"epoch {epoch}: {mean_acc}")

    torch.save(model.state_dict(), out_file)
