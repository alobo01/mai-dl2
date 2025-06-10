import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from deep_osr.models.backbone import get_backbone
from deep_osr.models.neck import EmbeddingNeck
from deep_osr.models.cls_head import ClassifierHead
from deep_osr.models.osr_head import EnergyOSRHead, KPlus1OSRHead, OpenMaxOSRHead
from deep_osr.losses.ce import get_loss_functions # Simpler: LabelSmoothingCrossEntropy directly
from deep_osr.open_set_metrics import OpenSetMetrics
from deep_osr.utils.calibration import calibrate_model_temperature

class OpenSetLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg) # Saves cfg to self.hparams
        self.cfg = cfg

        # 1. Backbone
        self.backbone, backbone_out_dim = get_backbone(
            name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            frozen=cfg.model.backbone.frozen
        )
        # Update config with actual backbone output dim if it was null
        if self.cfg.model.backbone.num_output_features is None:
             # This modification won't persist in the original OmegaConf object outside,
             # but self.hparams will have it. For consistency, ensure cfg is source of truth.
             # OmegaConf.set_struct(self.cfg.model.backbone, False) # Allow adding new keys
             # self.cfg.model.backbone.num_output_features = backbone_out_dim
             # OmegaConf.set_struct(self.cfg.model.backbone, True)
             # Better: pass backbone_out_dim directly to neck if cfg is tricky to update mid-init
             current_backbone_out_dim = backbone_out_dim
        else:
            current_backbone_out_dim = self.cfg.model.backbone.num_output_features


        # 2. Embedding Neck
        self.neck = EmbeddingNeck(
            in_features=current_backbone_out_dim,
            out_features=cfg.model.neck.out_features,
            use_batchnorm=cfg.model.neck.use_batchnorm,
            use_relu=cfg.model.neck.use_relu
        )

        # 3. Classifier Head (for K known classes)
        self.cls_head = ClassifierHead(
            in_features=cfg.model.d_embed, # d_embed is neck's out_features
            num_classes=cfg.dataset.num_known_classes,
            use_weight_norm=cfg.model.cls_head.use_weight_norm,
            temperature=1.0 # Temperature for logits scaling, 1.0 means no scaling during training here
                            # Calibration T applied post-training. Config T is for model's own scaling.
        )
        self.cls_head_train_temp = cfg.model.cls_head.temperature # Store configured temp

        # 4. Open-Set Recognition Head
        self.osr_head_type = cfg.model.osr_head.type
        if self.osr_head_type == "energy":
            self.osr_head = EnergyOSRHead(in_features=cfg.model.d_embed)
        elif self.osr_head_type == "kplus1":
            self.osr_head = KPlus1OSRHead(in_features=cfg.model.d_embed)
        elif self.osr_head_type == "openmax":
            self.osr_head = OpenMaxOSRHead(
                in_features=cfg.model.d_embed,
                num_known_classes=cfg.dataset.num_known_classes
            )
            # Note: OpenMax head needs `fit_mavs` to be called after initial training.
        else:
            raise ValueError(f"Unsupported OSR head type: {self.osr_head_type}")

        # 5. Loss functions
        # For simplicity, directly use LabelSmoothingCrossEntropy. Loss builder can be added for more complex scenarios.
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.model.cls_head.label_smoothing)
        self.loss_weights = {'ce_seen': cfg.train.loss.ce_seen_weight} # Add other loss weights if needed
        
        # 6. Metrics (using custom OpenSetMetrics class)
        self.train_metrics = OpenSetMetrics(prefix='train') # Not typical to run full OSR metrics on train
        self.val_metrics = OpenSetMetrics(prefix='val')
        self.test_metrics = OpenSetMetrics(prefix='test')

        # For storing things needed by eval.py, like calibration temperature
        self.calibration_temperature = 1.0 # Default, to be updated after calibration step

    def forward(self, x):
        """Defines the forward pass for inference/evaluation."""
        features = self.backbone(x)
        embeddings = self.neck(features)
        
        # Always get closed-set logits
        closed_set_logits_k = self.cls_head(embeddings) # Raw logits (temp not applied yet or T=1)

        osr_output = None
        if self.osr_head_type == "energy":
            # Energy score from the OSR head. Higher energy usually means more "in-distribution" by some conventions.
            # Here, blueprint "linear -> scalar energy". Let's assume higher energy = more unknown for consistency.
            # Or, typical: energy = -logsumexp(logits) -> lower energy = higher confidence = known
            # If osr_head is just linear -> scalar, then its sign/interpretation needs to be defined.
            # Let's assume self.osr_head(embeddings) gives a score where higher = more unknown.
            # For `EnergyOSRHead` linear layer, its weights will learn.
            # If we want E_out = - E_model, then `osr_score = -self.osr_head(embeddings)`
            # Let's make `EnergyOSRHead` learn to output high for unknowns.
            # The training objective will drive this. Here, just get the output.
            osr_score = self.osr_head(embeddings) # This is the direct output of the energy head.
            osr_output = osr_score 
        elif self.osr_head_type == "kplus1":
            unknown_logit = self.osr_head(embeddings) # (batch, 1)
            # For K+1, final_logits are (batch, K+1)
            final_logits_kplus1 = torch.cat([closed_set_logits_k, unknown_logit], dim=1)
            osr_output = final_logits_kplus1 # The OSR score will be derived from unknown_logit part
        elif self.osr_head_type == "openmax":
            # `osr_head` for OpenMax needs MAVs. It takes embeddings and K-class logits.
            # It returns (potentially revised K-class scores, unknown_score)
            # For now, it returns (original_K_logits, unknown_score_from_mav_dist)
            _, osr_score = self.osr_head(embeddings, closed_set_logits_k)
            osr_output = osr_score
        
        return closed_set_logits_k, osr_output

    def _get_osr_score_from_outputs(self, closed_set_logits_k, osr_head_output):
        """
        Standardizes OSR score: higher value means more likely unknown.
        closed_set_logits_k: (batch, K)
        osr_head_output: output from the OSR head logic in forward()
        """
        if self.osr_head_type == "energy":
            # Assume energy head is trained such that its direct output is the OSR score.
            # E.g. via a loss that pushes energy up for OOD, down for ID.
            # If no such loss, energy score's interpretation needs care.
            # Let's assume the raw output of EnergyOSRHead is the score.
            # To make it consistent (higher = more unknown), if it learns -logsumexp, then negate.
            # If it's a linear layer, its sign depends on training.
            # For now, let's use it directly. The loss for energy OSR head needs to be defined.
            # If only CE on knowns is used, then energy OSR head is trained just like any other layer.
            # Alternative common energy score: -torch.logsumexp(closed_set_logits_k, dim=1)
            # Let's use the direct output from the energy head.
            return osr_head_output

        elif self.osr_head_type == "kplus1":
            # osr_head_output is final_logits_kplus1 (batch, K+1)
            # OSR score can be probability of unknown class, or -logit_unknown
            # Let's use -logit_unknown. Higher value means more unknown.
            # (No, higher logit_unknown means more unknown. So, logit_unknown itself.)
            return osr_head_output[:, -1] # The last logit is for the unknown class

        elif self.osr_head_type == "openmax":
            # osr_head_output is already the unknown_score from OpenMax head (1 - max_similarity)
            return osr_head_output
        
        return torch.zeros(closed_set_logits_k.size(0), device=closed_set_logits_k.device)


    def _shared_step(self, batch, batch_idx, stage_metrics):
        x, y_true_known_idx, is_known_target = batch # y_true_known_idx is for K classes, only valid if is_known_target is True
        
        # Forward pass
        closed_set_logits_k, osr_head_raw_output = self(x) # Raw K-class logits (temp=1)
        
        # Apply training/configured temperature to K-class logits for CE loss
        closed_set_logits_k_scaled = closed_set_logits_k / self.cls_head_train_temp

        # --- Calculate Loss ---
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Standard CE loss for known samples, on K classes
        known_samples_mask = is_known_target
        if torch.any(known_samples_mask):
            loss_ce_seen = self.ce_loss_fn(
                closed_set_logits_k_scaled[known_samples_mask],
                y_true_known_idx[known_samples_mask]
            )
            total_loss += self.loss_weights['ce_seen'] * loss_ce_seen
            self.log(f'{stage_metrics.prefix}/loss_ce_seen', loss_ce_seen, on_step=False, on_epoch=True, prog_bar=False)

        # Additional losses based on OSR head type (if any defined beyond CE on knowns)
        # E.g., for energy-based, could add a term to push energy of knowns down, unknowns up.
        # E.g., for K+1, if training with outlier data, CE loss for (K+1)th class.
        # For this example, we primarily use CE on K knowns. OSR heads are either auxiliary
        # or their parameters are learned implicitly via this CE loss.

        self.log(f'{stage_metrics.prefix}/loss_total', total_loss, on_step=(stage_metrics.prefix=='train'), on_epoch=True, prog_bar=True)

        # --- For Metrics ---
        # Apply calibration temperature for evaluation logits if available (not during training loop directly)
        # For validation/test metrics during training, use the cls_head's configured temperature
        eval_temp = self.calibration_temperature if stage_metrics.prefix != 'train' and self.calibration_temperature != 1.0 else self.cls_head_train_temp
        
        probs_k_calibrated = F.softmax(closed_set_logits_k / eval_temp, dim=1)
        
        # Get standardized OSR score (higher = more unknown)
        osr_scores_standardized = self._get_osr_score_from_outputs(closed_set_logits_k, osr_head_raw_output)

        stage_metrics.update(probs_k_calibrated, osr_scores_standardized, y_true_known_idx, is_known_target)
        
        # For eval.py, we might want to return raw components for saving
        if stage_metrics.prefix == 'test': # Or a specific predict mode
            return closed_set_logits_k, osr_scores_standardized, y_true_known_idx, is_known_target, self.neck(self.backbone(x)) # also embeddings

        return total_loss

    def training_step(self, batch, batch_idx):
        # In training, we only have known samples as per OpenSetDataModule.setup()
        # So is_known_target should be all True.
        return self._shared_step(batch, batch_idx, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        # Validation set contains both known and unknown samples
        return self._shared_step(batch, batch_idx, self.val_metrics)

    def test_step(self, batch, batch_idx):
        # Test set structure depends on how test_dataloader is configured in DataModule
        # Assuming it can also provide mixed known/unknowns like val_dataloader
        return self._shared_step(batch, batch_idx, self.test_metrics)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(num_known_classes=self.cfg.dataset.num_known_classes)
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def on_train_epoch_end(self): # Typically not needed unless specific train OSR metrics
        # metrics = self.train_metrics.compute(num_known_classes=self.cfg.dataset.num_known_classes)
        # self.log_dict(metrics)
        self.train_metrics.reset()


    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(num_known_classes=self.cfg.dataset.num_known_classes)
        self.log_dict(metrics)
        self.test_metrics.reset()
        # `eval.py` will handle detailed metric saving. This is for PL's default test reporting.

    def configure_optimizers(self):
        # Filter out frozen parameters
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        if self.cfg.train.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
        elif self.cfg.train.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay)
        elif self.cfg.train.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(trainable_params, lr=self.cfg.train.optimizer.lr, weight_decay=self.cfg.train.optimizer.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.cfg.train.optimizer.name} not supported.")

        if self.cfg.train.scheduler and self.cfg.train.scheduler.name:
            if self.cfg.train.scheduler.name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.cfg.train.scheduler.params)
            elif self.cfg.train.scheduler.name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.cfg.train.scheduler.params)
            else:
                raise ValueError(f"Scheduler {self.cfg.train.scheduler.name} not supported.")
            return [optimizer], [scheduler]
        
        return optimizer

    # Hook for OpenMax MAV fitting (called after training finishes)
    def fit_openmax_if_needed(self, datamodule):
        if self.osr_head_type == "openmax" and isinstance(self.osr_head, OpenMaxOSRHead):
            # Use training dataloader (knowns only) to fit MAVs
            # Need a feature extractor function that gives embeddings
            def feature_extractor_fn(images_batch):
                return self.neck(self.backbone(images_batch))
            
            # Get the training dataloader that yields (images, labels_known_idx, is_known_mask=True)
            # The OpenSetDataModule's train_dataloader() is suitable.
            train_dl_for_mav = datamodule.train_dataloader()
            self.osr_head.fit_mavs(train_dl_for_mav, feature_extractor_fn, self.device)
            print("OpenMax MAVs fitted.")
        else:
            print("OpenMax head not configured or MAVs not applicable.")
            
    # Hook for temperature calibration (called after training, before final eval)
    def calibrate_temperature(self, datamodule):
        if self.cfg.train.calibration.method == "temperature_scaling":
            print("Calibrating temperature...")
            # Collect model outputs on calibration set (e.g., validation set)
            # Similar to `validation_step` but without gradient tracking etc.
            calib_dl = datamodule.calibration_dataloader() # Assumes this provides (img, label_idx, is_known)
            
            all_logits_k_batches = []
            all_osr_score_batches = [] # Not used for temp scaling but collected for completeness
            all_y_true_batches = []
            all_is_known_mask_batches = []

            self.eval() # Set model to evaluation mode
            with torch.no_grad():
                for batch in calib_dl:
                    x, y_true_known_idx, is_known_target = batch
                    x = x.to(self.device)
                    
                    closed_set_logits_k, osr_head_raw_output = self(x) # Forward pass
                    osr_scores_standardized = self._get_osr_score_from_outputs(closed_set_logits_k, osr_head_raw_output)

                    all_logits_k_batches.append(closed_set_logits_k.cpu())
                    all_osr_score_batches.append(osr_scores_standardized.cpu())
                    all_y_true_batches.append(y_true_known_idx.cpu())
                    all_is_known_mask_batches.append(is_known_target.cpu())
            
            model_outputs_val = list(zip(all_logits_k_batches, all_osr_score_batches))
            labels_val = list(zip(all_y_true_batches, all_is_known_mask_batches))

            optimal_t = calibrate_model_temperature(model_outputs_val, labels_val, self.device)
            self.calibration_temperature = optimal_t
            print(f"Set model calibration temperature to: {self.calibration_temperature:.4f}")
        else:
            print("No calibration method specified or temperature scaling not selected.")