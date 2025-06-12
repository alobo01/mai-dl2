import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from deep_osr.models.backbone import get_backbone
from deep_osr.models.neck import EmbeddingNeck
from deep_osr.models.cls_head import ClassifierHead
from deep_osr.models.osr_head import EnergyOSRHead, KPlus1OSRHead, OpenMaxOSRHead
from deep_osr.losses.enhanced_nll import get_enhanced_loss_functions
from deep_osr.open_set_metrics import OpenSetMetrics
from deep_osr.utils.calibration import calibrate_model_temperature


class EnhancedOpenSetLightningModule(pl.LightningModule):
    """
    Enhanced Lightning Module for open-set recognition with dummy class support.
    Supports various loss strategies including thresholding and penalty-based approaches.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # self.save_hyperparameters(cfg) # Old way, inherited and would be problematic
        # The call in the parent class OpenSetLightningModule's __init__ 
        # already does self.save_hyperparameters() which correctly saves {'cfg': cfg_object}.
        # No explicit call to save_hyperparameters is needed here if the parent does it correctly.

        # Ensure enhanced loss functions are initialized
        self.cfg = cfg
        self.save_hyperparameters()

        # 1. Backbone
        self.backbone, backbone_out_dim = get_backbone(
            name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            frozen=cfg.model.backbone.frozen
        )
        
        current_backbone_out_dim = backbone_out_dim if cfg.model.backbone.num_output_features is None else cfg.model.backbone.num_output_features

        # 2. Embedding Neck (optional)
        if cfg.model.get('neck') and cfg.model.neck.get('enabled', True):
            self.neck = EmbeddingNeck(
                in_features=current_backbone_out_dim,
                out_features=cfg.model.neck.out_features,
                use_batchnorm=cfg.model.neck.use_batchnorm,
                use_relu=cfg.model.neck.use_relu
            )
            cls_in_features = cfg.model.neck.out_features
        else:
            self.neck = nn.Identity()
            cls_in_features = current_backbone_out_dim

        # 3. Classifier Head for K known classes
        self.cls_head = ClassifierHead(
            in_features=cls_in_features,
            num_classes=cfg.dataset.num_known_classes,
            use_weight_norm=cfg.model.cls_head.use_weight_norm,
            temperature=1.0
        )
        self.cls_head_train_temp = cfg.model.cls_head.temperature

        # 4. Dummy Class Head (K+1 classes including dummy)
        if cfg.train.loss.get('use_dummy_class', False):
            self.dummy_cls_head = ClassifierHead(
                in_features=cls_in_features,
                num_classes=cfg.dataset.num_known_classes + 1,  # +1 for dummy class
                use_weight_norm=cfg.model.cls_head.use_weight_norm,
                temperature=1.0
            )
        else:
            self.dummy_cls_head = None

        # 5. OSR Head (optional)
        self.osr_head_type = cfg.model.get('osr_head', {}).get('type', 'none')
        if self.osr_head_type == "energy":
            self.osr_head = EnergyOSRHead(in_features=cls_in_features)
        elif self.osr_head_type == "kplus1":
            self.osr_head = KPlus1OSRHead(in_features=cls_in_features)
        elif self.osr_head_type == "openmax":
            self.osr_head = OpenMaxOSRHead(
                in_features=cls_in_features,
                num_known_classes=cfg.dataset.num_known_classes
            )
        else:
            self.osr_head = None

        # 6. Loss functions
        self.loss_strategy = cfg.train.loss.strategy
        
        if self.loss_strategy == "standard":
            # Standard cross-entropy
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.model.cls_head.label_smoothing)
        else:
            # Enhanced loss functions
            loss_kwargs = {
                'confidence_threshold': cfg.train.loss.get('confidence_threshold', 0.5),
                'dummy_class_penalty': cfg.train.loss.get('dummy_class_penalty', 2.0),
                'smoothing': cfg.model.cls_head.label_smoothing,
                'threshold_weight': cfg.train.loss.get('threshold_weight', 0.5),
                'penalty_weight': cfg.train.loss.get('penalty_weight', 0.5),
            }
            self.loss_fn = get_enhanced_loss_functions(self.loss_strategy, **loss_kwargs)

        # 7. Loss weights
        self.loss_weights = {
            'ce_seen': cfg.train.loss.get('ce_seen_weight', 1.0),
            'dummy_loss': cfg.train.loss.get('dummy_loss_weight', 1.0),
            'osr_loss': cfg.train.loss.get('osr_loss_weight', 0.1),
        }

        # 8. Metrics
        self.train_metrics = OpenSetMetrics(prefix='train')
        self.val_metrics = OpenSetMetrics(prefix='val')
        self.test_metrics = OpenSetMetrics(prefix='test')

        # 9. Calibration
        self.calibration_temperature = 1.0

    def forward(self, x):
        """Forward pass for inference/evaluation."""
        features = self.backbone(x)
        embeddings = self.neck(features)
        
        # Always get closed-set logits for K classes
        closed_set_logits_k = self.cls_head(embeddings)
        
        # Get dummy class logits if available
        dummy_logits = None
        if self.dummy_cls_head is not None:
            dummy_logits = self.dummy_cls_head(embeddings)
        
        # Get OSR output if available
        osr_output = None
        if self.osr_head is not None:
            if self.osr_head_type == "energy":
                osr_output = self.osr_head(embeddings)
            elif self.osr_head_type == "kplus1":
                osr_output = self.osr_head(embeddings)
            elif self.osr_head_type == "openmax":
                _, osr_output = self.osr_head(embeddings, closed_set_logits_k)
        
        return closed_set_logits_k, dummy_logits, osr_output

    def _get_osr_score_from_outputs(self, closed_set_logits_k, dummy_logits, osr_output):
        """Standardizes OSR score: higher value means more likely unknown."""
        if self.osr_head_type == "energy":
            return osr_output if osr_output is not None else torch.zeros(closed_set_logits_k.size(0), device=closed_set_logits_k.device)
        elif self.osr_head_type == "kplus1":
            return osr_output[:, -1] if osr_output is not None else torch.zeros(closed_set_logits_k.size(0), device=closed_set_logits_k.device)
        elif self.osr_head_type == "openmax":
            return osr_output if osr_output is not None else torch.zeros(closed_set_logits_k.size(0), device=closed_set_logits_k.device)
        elif dummy_logits is not None:
            # Use dummy class logit as OSR score
            return dummy_logits[:, -1]  # Last class is dummy
        else:
            # Use negative max confidence as OSR score
            probs = F.softmax(closed_set_logits_k, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            return 1.0 - max_probs

    def _prepare_targets_for_dummy_class(self, y_true_known_idx, is_known_target, num_known_classes):
        """Prepare targets for dummy class training."""
        # For known samples: use original labels (0 to K-1)
        # For unknown samples: use dummy class label (K)
        targets_with_dummy = y_true_known_idx.clone()
        targets_with_dummy[~is_known_target] = num_known_classes  # Dummy class index
        return targets_with_dummy

    def _shared_step(self, batch, batch_idx, stage_metrics):
        x, y_true_known_idx, is_known_target = batch
        
        # Forward pass
        closed_set_logits_k, dummy_logits, osr_output = self(x)
        
        # Scale logits for loss computation
        closed_set_logits_k_scaled = closed_set_logits_k / self.cls_head_train_temp
        
        # Calculate loss
        total_loss = torch.tensor(0.0, device=self.device)
        
        if self.loss_strategy == "standard":
            # Standard training: only use known samples
            known_samples_mask = is_known_target
            if torch.any(known_samples_mask):
                loss_ce_seen = self.loss_fn(
                    closed_set_logits_k_scaled[known_samples_mask],
                    y_true_known_idx[known_samples_mask]
                )
                total_loss += self.loss_weights['ce_seen'] * loss_ce_seen
                
        elif self.loss_strategy == "threshold":
            # Thresholding approach: use all samples
            known_samples_mask = is_known_target
            if torch.any(known_samples_mask):
                loss_threshold = self.loss_fn(
                    closed_set_logits_k_scaled[known_samples_mask],
                    y_true_known_idx[known_samples_mask],
                    known_samples_mask[known_samples_mask]
                )
                total_loss += self.loss_weights['ce_seen'] * loss_threshold
                
        elif self.loss_strategy == "penalty":
            # Penalty approach: use dummy class head
            if dummy_logits is not None:
                targets_with_dummy = self._prepare_targets_for_dummy_class(
                    y_true_known_idx, is_known_target, self.cfg.dataset.num_known_classes
                )
                dummy_logits_scaled = dummy_logits / self.cls_head_train_temp
                loss_penalty = self.loss_fn(dummy_logits_scaled, targets_with_dummy, is_known_target)
                total_loss += self.loss_weights['dummy_loss'] * loss_penalty
                
        elif self.loss_strategy == "combined":
            # Combined approach: use both strategies
            known_samples_mask = is_known_target
            if torch.any(known_samples_mask) and dummy_logits is not None:
                targets_with_dummy = self._prepare_targets_for_dummy_class(
                    y_true_known_idx, is_known_target, self.cfg.dataset.num_known_classes
                )
                dummy_logits_scaled = dummy_logits / self.cls_head_train_temp
                
                loss_combined = self.loss_fn(
                    closed_set_logits_k_scaled[known_samples_mask],
                    dummy_logits_scaled,
                    y_true_known_idx[known_samples_mask],
                    targets_with_dummy,
                    is_known_target
                )
                total_loss += self.loss_weights['ce_seen'] * loss_combined

        # Additional OSR loss if available
        if self.osr_head is not None and osr_output is not None:
            # Simple energy regularization: push energy up for unknowns, down for knowns
            if self.osr_head_type == "energy":
                energy_targets = (~is_known_target).float()  # 1 for unknown, 0 for known
                osr_loss = F.mse_loss(torch.sigmoid(osr_output), energy_targets)
                total_loss += self.loss_weights['osr_loss'] * osr_loss

        self.log(f'{stage_metrics.prefix}/loss_total', total_loss, 
                on_step=(stage_metrics.prefix=='train'), on_epoch=True, prog_bar=True)

        # Metrics computation
        eval_temp = self.calibration_temperature if stage_metrics.prefix != 'train' and self.calibration_temperature != 1.0 else self.cls_head_train_temp
        probs_k_calibrated = F.softmax(closed_set_logits_k / eval_temp, dim=1)
        osr_scores_standardized = self._get_osr_score_from_outputs(closed_set_logits_k, dummy_logits, osr_output)
        
        stage_metrics.update(probs_k_calibrated, osr_scores_standardized, y_true_known_idx, is_known_target)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.test_metrics)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(num_known_classes=self.cfg.dataset.num_known_classes)
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(num_known_classes=self.cfg.dataset.num_known_classes)
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        if self.cfg.train.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.train.optimizer.lr, 
                                       weight_decay=self.cfg.train.optimizer.weight_decay)
        elif self.cfg.train.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.train.optimizer.lr, 
                                        weight_decay=self.cfg.train.optimizer.weight_decay)
        elif self.cfg.train.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(trainable_params, lr=self.cfg.train.optimizer.lr, 
                                      weight_decay=self.cfg.train.optimizer.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.cfg.train.optimizer.name} not supported.")

        if self.cfg.train.get('scheduler') and self.cfg.train.scheduler.get('name'):
            if self.cfg.train.scheduler.name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.cfg.train.scheduler.params)
            elif self.cfg.train.scheduler.name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.cfg.train.scheduler.params)
            else:
                raise ValueError(f"Scheduler {self.cfg.train.scheduler.name} not supported.")
            return [optimizer], [scheduler]
        
        return optimizer

    def fit_openmax_if_needed(self, datamodule):
        if self.osr_head_type == "openmax" and isinstance(self.osr_head, OpenMaxOSRHead):
            def feature_extractor_fn(images_batch):
                return self.neck(self.backbone(images_batch))
            
            train_dl_for_mav = datamodule.train_dataloader()
            self.osr_head.fit_mavs(train_dl_for_mav, feature_extractor_fn, self.device)
            print("OpenMax MAVs fitted.")
        else:
            print("OpenMax head not configured or MAVs not applicable.")

    def calibrate_temperature(self, datamodule):
        if self.cfg.train.get('calibration', {}).get('method') == "temperature_scaling":
            print("Calibrating temperature...")
            calib_dl = datamodule.calibration_dataloader()
            all_logits_k_batches = []
            all_osr_scores_batches = []
            all_y_true_batches = []
            all_is_known_mask_batches = []

            self.eval()
            with torch.no_grad():
                for batch in calib_dl:
                    x, y_true_known_idx, is_known_target = batch
                    x = x.to(self.device)
                    
                    closed_set_logits_k, dummy_logits, osr_output = self(x)
                    osr_scores = self._get_osr_score_from_outputs(closed_set_logits_k, dummy_logits, osr_output)
                    
                    all_logits_k_batches.append(closed_set_logits_k.cpu())
                    all_osr_scores_batches.append(osr_scores.cpu())
                    all_y_true_batches.append(y_true_known_idx.cpu())
                    all_is_known_mask_batches.append(is_known_target.cpu())
            
            model_outputs_val = list(zip(all_logits_k_batches, all_osr_scores_batches))
            labels_val = list(zip(all_y_true_batches, all_is_known_mask_batches))

            optimal_t = calibrate_model_temperature(model_outputs_val, labels_val, self.device)
            self.calibration_temperature = optimal_t
            print(f"Set model calibration temperature to: {self.calibration_temperature:.4f}")
        else:
            print("No calibration method specified or temperature scaling not selected.")
