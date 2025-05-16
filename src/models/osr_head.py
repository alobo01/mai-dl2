import torch
import torch.nn as nn
# For OpenMax, we might need a way to fit distributions (e.g., libMR from original OpenMax)
# For this example, OpenMax will be simplified or use placeholder logic.

class EnergyOSRHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear_to_energy = nn.Linear(in_features, 1)

    def forward(self, embeddings):
        # embeddings: (batch_size, d_embed)
        # Output: (batch_size, 1) scalar energy value
        energy = self.linear_to_energy(embeddings)
        return energy.squeeze(-1) # Return as (batch_size,)

class KPlus1OSRHead(nn.Module):
    """ Outputs a single logit for the K+1-th (unknown) class. """
    def __init__(self, in_features: int):
        super().__init__()
        self.unknown_logit_layer = nn.Linear(in_features, 1)

    def forward(self, embeddings):
        # embeddings: (batch_size, d_embed)
        # Output: (batch_size, 1) logit for the unknown class
        unknown_logit = self.unknown_logit_layer(embeddings)
        return unknown_logit

class OpenMaxOSRHead(nn.Module):
    """
    Simplified OpenMax: uses distances to class centroids (MAVs).
    Full OpenMax involves Weibull fitting, which is complex.
    This head will require MAVs to be computed and set after initial model training.
    """
    def __init__(self, in_features: int, num_known_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_known_classes = num_known_classes
        
        # MAVs will be registered as buffers after fitting
        self.register_buffer("mavs", torch.zeros(num_known_classes, in_features))
        self.fitted = False # Flag to check if MAVs are computed

    def fit_mavs(self, dataloader, feature_extractor_fn, device):
        """
        Computes Mean Activation Vectors (MAVs) for each known class.
        feature_extractor_fn: a function that takes a batch of images and returns embeddings.
        """
        print("Fitting MAVs for OpenMax head...")
        all_features = [[] for _ in range(self.num_known_classes)]
        
        for batch in dataloader: # Assuming dataloader yields (images, labels_known_idx, is_known_mask)
            images, labels_known_idx, is_known_mask = batch
            images = images.to(device)
            
            known_images = images[is_known_mask]
            known_labels = labels_known_idx[is_known_mask]

            if len(known_images) == 0:
                continue

            with torch.no_grad():
                features = feature_extractor_fn(known_images) # (num_known_in_batch, d_embed)
            
            for i in range(self.num_known_classes):
                class_features = features[known_labels == i]
                if len(class_features) > 0:
                    all_features[i].append(class_features.cpu())
        
        mavs_list = []
        for i in range(self.num_known_classes):
            if len(all_features[i]) > 0:
                class_features_all = torch.cat(all_features[i], dim=0)
                mavs_list.append(class_features_all.mean(dim=0))
            else:
                # Handle classes with no samples in the fitting data (e.g., use zero vector or raise error)
                print(f"Warning: No samples found for class {i} during MAV fitting. Using zero vector.")
                mavs_list.append(torch.zeros(self.in_features))
        
        self.mavs = torch.stack(mavs_list).to(device)
        self.fitted = True
        print("MAVs fitting complete.")

    def forward(self, embeddings, closed_set_logits_k):
        """
        embeddings: (batch_size, d_embed)
        closed_set_logits_k: (batch_size, K) logits for known classes
        Returns:
            openmax_probs: (batch_size, K+1) probabilities including an unknown class.
                           For simplicity, this might just be K softmax scores and a separate unknown score.
            unknown_scores: (batch_size,) score indicating "unknownness". Higher = more unknown.
        """
        if not self.fitted:
            raise RuntimeError("OpenMax head MAVs not fitted. Call fit_mavs() first.")

        # Calculate distances to MAVs (e.g., Euclidean)
        # dists = torch.cdist(embeddings, self.mavs)  # (batch_size, num_known_classes)
        
        # Simplified: Use cosine similarity (negative distance) as part of features for unknown score
        # A common way to get an "unknown" score with MAVs is min distance or similar.
        # For a basic implementation, let's consider the largest activation if using softmax on distances.
        # Or, max logit of known classes MINUS distance to corresponding MAV.
        # This is a placeholder for real OpenMax.
        # A simple unknown score: min distance to any MAV. Small distance = more likely known.
        # So, unknown_score = min_dist. To make higher = more unknown, this is fine.
        
        # Let's compute cosine similarities. Higher similarity means more "known-like".
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), self.mavs.unsqueeze(0), dim=2) # (batch, K)
        
        # An unknown score could be 1 - max_similarity (so higher = more unknown)
        max_similarity_to_mavs, _ = torch.max(similarities, dim=1) # (batch_size)
        unknown_scores = 1.0 - max_similarity_to_mavs 

        # For OpenMax, one would typically revise the K logits based on Weibull fits of distances
        # and compute a K+1-th probability for unknown.
        # Here, we just return the original K logits and our simple unknown_score.
        # The `open_set_metrics` will use `unknown_scores` for AUROC etc.
        # `closed_set_logits_k` can be used for closed-set accuracy on known samples.
        return closed_set_logits_k, unknown_scores