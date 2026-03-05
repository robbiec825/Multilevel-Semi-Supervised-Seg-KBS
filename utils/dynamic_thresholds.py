import math
import  numpy as np
from scipy import stats
from collections import defaultdict
import torch



class DynamicConfidenceThreshold_max_percentile:
    def __init__(self, alpha=0.9, percentile=95, labeled_ratio = 0.1, num_classes=4):
        
        self.alpha = alpha
        self.percentile = percentile
        self.initialized = False
        self.class_top_means = {}
        self.best_class_threshold = None
        self.class_threshold_adjustments = {}
        self.current_pred_logits = None
        self.historical_best_means = {}
        self.performance_history = defaultdict(list)
        self.labeled_ratio = labeled_ratio
        self.num_classes = num_classes


    def collect_max_predictions_by_class(self, model_outputs_list, num_classes):
        class_predictions = {cls: [] for cls in range(num_classes)}
        with torch.no_grad():
            for outputs in model_outputs_list:
                max_probs, pred_classes = torch.max(outputs, dim=1)
                for cls in range(num_classes):
                    mask = (pred_classes == cls)
                    if mask.sum() > 0:
                        class_predictions[cls].append(max_probs[mask].view(-1))
        for cls in range(num_classes):
            if len(class_predictions[cls]) > 0:
                class_predictions[cls] = torch.cat(class_predictions[cls])
            else:
                class_predictions[cls] = torch.tensor([], device=model_outputs_list[0].device)
        return class_predictions

    def update_statistics(self, model_outputs_list):
        self.current_pred_logits = model_outputs_list
        class_predictions = self.collect_max_predictions_by_class(model_outputs_list, self.num_classes)
        current_top_means = {}

        for cls in range(self.num_classes):
            if class_predictions[cls].numel() > 0:
                cls_probs = class_predictions[cls]
                if cls_probs.numel() > 20:
                    threshold = torch.quantile(cls_probs, self.percentile / 100.0)
                    top_samples = cls_probs[cls_probs >= threshold]
                    if top_samples.numel() > 0:
                        current_top_means[cls] = top_samples.mean().item()
                else:
                    current_top_means[cls] = cls_probs.max().item()   
        

        if not current_top_means:
            return 

        best_class = max(current_top_means.items(), key=lambda x: x[1])[0]
        best_mean = current_top_means[best_class]

        for cls, mean in current_top_means.items():
            if cls not in self.historical_best_means:
                self.historical_best_means[cls] = mean
            else:
                self.historical_best_means[cls] = max(
                    self.historical_best_means[cls],
                    self.alpha * self.historical_best_means[cls] + (1 - self.alpha) * mean
                )

        relative_improvements = {}
        for cls, mean in current_top_means.items():
            hist_best = self.historical_best_means.get(cls, mean)
            relative_improvements[cls] = mean / hist_best if hist_best > 0 else 0.5

        for cls, mean in current_top_means.items():
            self.performance_history[cls].append(mean)
  
        all_means = list(current_top_means.values())
        
        all_std_devs = []
        for c, mean_list in self.performance_history.items():
            if len(mean_list) > 1:  
                c_std = np.std(mean_list)
                all_std_devs.append(c_std)
        mean_of_std_devs = np.mean(all_std_devs) if all_std_devs else 1.0
      
        current_adjustments = {}
        for cls, mean in current_top_means.items():
            adjustment = math.log(mean + 1) / math.log( 2 + math.log(1/self.labeled_ratio)/math.log(100)*0.2)
            
            current_adjustments[cls] = adjustment

        self.class_top_means = current_top_means
        self.class_threshold_adjustments = current_adjustments

        if not self.initialized:
            self.best_class_threshold = best_mean
            self.initialized = True
        else:
            self.best_class_threshold = self.alpha * self.best_class_threshold + (1 - self.alpha) * best_mean

    def sigmoid_mapping(self, x, k=10, x0=0.7, min_val=0.4, max_val=0.90):
        sigmoid = 1 / (1 + math.exp(-k * (x - x0)))
        return min_val + sigmoid * (max_val - min_val)

    def get_weight(self, class_id):
        if not self.initialized or class_id not in self.class_threshold_adjustments:
            return 1 / self.num_classes
        threshold = self.best_class_threshold * self.class_threshold_adjustments[class_id]
        return max(threshold, 1 / self.num_classes)
    







