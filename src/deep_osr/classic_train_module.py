import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import torchmetrics

from deep_osr.models.backbone import get_backbone
from deep_osr.models.neck import EmbeddingNeck
from deep_osr.models.cls_head import ClassifierHead # Reusing the same classifier head

class ClassicLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({'cfg': cfg})

        # 1. Backbone
        self.backbone, backbone_out_dim = get_backbone(
            name=cfg.model.backbone.name,
            pretrained=cfg.model.backbone.pretrained,
            frozen=cfg.model.backbone.frozen
        )
        current_backbone_out_dim = backbone_out_dim if cfg.model.backbone.num_output_features is None else cfg.model.backbone.num_output_features

        # 2. Embedding Neck (optional, can be identity if not needed for classic)
        if cfg.model.get('neck'):
            self.neck = EmbeddingNeck(
                in_features=current_backbone_out_dim,
                out_features=cfg.model.neck.out_features,
                use_batchnorm=cfg.model.neck.use_batchnorm,
                use_relu=cfg.model.neck.use_relu
            )
            cls_in_features = cfg.model.neck.out_features
        else: # No neck, backbone output goes directly to classifier
            self.neck = nn.Identity()
            cls_in_features = current_backbone_out_dim

        # 3. Classifier Head
        self.cls_head = ClassifierHead(
            in_features=cls_in_features, 
            num_classes=cfg.dataset.num_classes, # Total number of classes for classic classification
            use_weight_norm=cfg.model.cls_head.use_weight_norm,
            temperature=cfg.model.cls_head.get('temperature', 1.0)
        )

        # 4. Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.model.cls_head.get('label_smoothing', 0.0))

        # 5. Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average='macro')


    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        logits = self.cls_head(embeddings)
        return logits

    def _shared_step(self, batch, batch_idx):
        x, y = batch # Classic datamodule should return (image, label)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.train_acc(preds, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.val_acc(preds, y)
        self.val_f1(preds,y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1_macro', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.test_acc(preds, y)
        self.test_f1(preds,y)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1_macro', self.test_f1, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer_cfg = self.cfg.train.optimizer
        if optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(trainable_params, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.get('weight_decay', 0.0))
        elif optimizer_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.get('weight_decay', 0.0001))
        elif optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(trainable_params, lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.get('weight_decay', 0.0), momentum=optimizer_cfg.get('momentum', 0.9))
        else:
            raise ValueError(f"Optimizer {optimizer_cfg.name} not supported.")

        if self.cfg.train.get('scheduler') and self.cfg.train.scheduler.get('name'):
            scheduler_cfg = self.cfg.train.scheduler
            if scheduler_cfg.name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg.params)
            elif scheduler_cfg.name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg.params)
            else:
                raise ValueError(f"Scheduler {scheduler_cfg.name} not supported.")
            return [optimizer], [scheduler]
        
        return optimizer
