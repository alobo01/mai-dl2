# Enhanced Open-Set Recognition Experiments

This repository implements enhanced open-set recognition experiments using modified Negative Log-Likelihood (NLL) loss functions with dummy class training approaches.

## Overview

The enhanced implementation introduces three novel loss strategies for training OSR models with dummy classes:

1. **Threshold-based Loss**: Uses confidence thresholding to identify and penalize low-confidence predictions
2. **Penalty-based Loss**: Applies higher penalties for dummy class predictions 
3. **Combined Loss**: Combines both threshold and penalty approaches with configurable weights

## Key Features

- **Enhanced Loss Functions**: Three specialized loss functions with configurable parameters
- **Dummy Class Training**: Training with unknown samples as dummy classes
- **Comprehensive Experiments**: 12 pre-configured experiments across CIFAR-10 and GTSRB
- **Parallel Execution**: Support for multi-GPU parallel training
- **Detailed Analysis**: Comprehensive evaluation and comparison tools

## Project Structure

```
DL2/
├── src/
│   ├── deep_osr/
│   │   ├── losses/
│   │   │   └── enhanced_nll.py          # Enhanced loss functions
│   │   ├── enhanced_train_module.py     # Enhanced Lightning module
│   │   └── enhanced_data_module.py      # Enhanced data module
│   ├── eval.py                          # Evaluation script
│   └── train.py                         # Training script
├── configs/
│   ├── model/
│   │   └── resnet50_enhanced_osr.yaml   # Enhanced model config
│   ├── train/
│   │   └── enhanced_osr.yaml            # Enhanced training config
│   ├── exp_cifar10_*.yaml               # CIFAR-10 experiments
│   └── exp_gtsrb_*.yaml                 # GTSRB experiments
├── run_enhanced_experiments.py          # Main experiment runner
├── analyze_enhanced_results.py          # Results analysis
└── test_enhanced_implementation.py      # Implementation validator
```

## Enhanced Loss Functions

### 1. Threshold-based NLL Loss (`ThresholdedNLLLoss`)

Applies penalties to predictions below a confidence threshold:

```python
penalty = torch.where(
    max_probs < self.confidence_threshold,
    -torch.log(max_probs + 1e-8),
    torch.zeros_like(max_probs)
)
loss = standard_nll + penalty
```

**Parameters:**
- `confidence_threshold`: Threshold for low-confidence penalty (0.4-0.8)

### 2. Penalty-based NLL Loss (`DummyClassNLLLoss`)

Applies higher penalties for dummy class predictions:

```python
dummy_penalty = self.dummy_penalty_factor * nll_loss
loss = torch.where(is_dummy, dummy_penalty, nll_loss)
```

**Parameters:**
- `dummy_penalty_factor`: Penalty multiplier for dummy classes (1.5-4.5)

### 3. Combined Threshold-Penalty Loss (`CombinedThresholdPenaltyLoss`)

Combines both approaches with configurable weights:

```python
loss = (self.threshold_weight * threshold_loss + 
        self.penalty_weight * penalty_loss)
```

**Parameters:**
- `confidence_threshold`: Threshold for low-confidence penalty
- `dummy_penalty_factor`: Penalty multiplier for dummy classes
- `threshold_weight`: Weight for threshold component (0.3-0.7)
- `penalty_weight`: Weight for penalty component (0.3-0.7)

## Experiment Configurations

### CIFAR-10 Experiments

| Experiment | Strategy | Unknown Classes | Architecture | Key Parameters |
|------------|----------|----------------|--------------|----------------|
| `threshold_1u` | Threshold | 1 | No Neck | threshold=0.6 |
| `penalty_3u` | Penalty | 3 | With Neck | penalty=2.0 |
| `combined_5u` | Combined | 5 | No Neck | threshold=0.5, penalty=2.5 |
| `high_penalty_1u` | Penalty | 1 | With Neck | penalty=4.0 |
| `high_threshold_3u` | Threshold | 3 | No Neck | threshold=0.8 |
| `optimized_combined_5u` | Combined | 5 | With Neck | threshold=0.4, penalty=3.0 |

### GTSRB Experiments

| Experiment | Strategy | Unknown Classes | Architecture | Key Parameters |
|------------|----------|----------------|--------------|----------------|
| `threshold_3u` | Threshold | 3 | No Neck | threshold=0.6 |
| `penalty_6u` | Penalty | 6 | With Neck | penalty=2.5 |
| `combined_9u` | Combined | 9 | No Neck | threshold=0.5, penalty=3.0 |
| `high_penalty_3u` | Penalty | 3 | With Neck | penalty=4.5 |
| `low_threshold_6u` | Threshold | 6 | No Neck | threshold=0.4 |
| `optimized_combined_9u` | Combined | 9 | With Neck | threshold=0.6, penalty=2.0 |

## Installation and Setup

1. **Install Dependencies**:
```bash
pip install torch torchvision pytorch-lightning
pip install hydra-core wandb
pip install scikit-learn matplotlib seaborn pandas
```

2. **Validate Implementation**:
```bash
python test_enhanced_implementation.py
```

## Usage

### Running Experiments

1. **Single Dataset**:
```bash
python run_enhanced_experiments.py --dataset cifar10
python run_enhanced_experiments.py --dataset gtsrb
```

2. **All Experiments**:
```bash
python run_enhanced_experiments.py --dataset all
```

3. **Parallel Execution**:
```bash
python run_enhanced_experiments.py --parallel --gpu-ids 0,1,2,3
```

4. **Custom Configuration**:
```bash
python run_enhanced_experiments.py --config-dir custom_configs
```

### Analyzing Results

1. **Generate Analysis**:
```bash
python analyze_enhanced_results.py
```

2. **Custom Analysis**:
```bash
python analyze_enhanced_results.py --results-dir custom_results --output-dir custom_analysis
```

### Manual Training

For individual experiments:

```bash
python -m src.train --config-path configs --config-name exp_cifar10_threshold_1u
```

## Implementation Details

### Enhanced Training Module

The `EnhancedOpenSetLightningModule` extends the base Lightning module with:

- **Dummy Class Support**: Optional dummy classifier head
- **Multiple Loss Strategies**: Configurable loss function selection
- **Enhanced Forward Pass**: Unified output format with OSR scores
- **Flexible Architecture**: Optional neck layer configuration

### Enhanced Data Module

The `EnhancedOpenSetDataModule` provides:

- **Unknown Sample Integration**: Includes unknown samples in training
- **Configurable Ratios**: Adjustable unknown/known sample ratios
- **Dummy Class Labeling**: Automatic labeling for dummy class training
- **Balanced Sampling**: Ensures balanced training batches

### Enhanced Loss Functions

All loss functions support:

- **Configurable Parameters**: Threshold and penalty values
- **Gradient Stability**: Numerical stability improvements
- **Unknown Sample Handling**: Proper handling of dummy class samples
- **Flexible Integration**: Easy integration with existing training loops

## Evaluation Metrics

The experiments track multiple metrics:

- **Known Accuracy**: Accuracy on seen classes
- **Unknown Accuracy**: Accuracy on unseen classes (detection rate)
- **AUROC**: Area under ROC curve for OSR task
- **AUPR**: Area under precision-recall curve
- **FPR@95**: False positive rate at 95% true positive rate
- **Training Metrics**: Loss convergence, best epochs, etc.

## Results and Analysis

The analysis pipeline generates:

1. **Performance Comparisons**: Strategy and dataset comparisons
2. **Parameter Sensitivity**: Impact of threshold and penalty values
3. **Architecture Analysis**: With/without neck comparisons
4. **Convergence Studies**: Training dynamics and stability
5. **Statistical Reports**: Detailed performance statistics

### Key Findings

Based on the experimental design:

- **Combined strategies** often outperform single approaches
- **Parameter tuning** significantly impacts performance
- **Architecture choices** matter for different datasets
- **Unknown class count** affects optimal strategy selection

## Customization

### Adding New Loss Functions

1. Inherit from `torch.nn.Module`
2. Implement `forward()` method
3. Add to loss function registry
4. Update configuration files

### Adding New Experiments

1. Create new YAML configuration
2. Specify loss strategy and parameters
3. Configure dataset and model settings
4. Add to experiment runner

### Extending Analysis

1. Add new metrics to evaluation script
2. Extend analysis functions
3. Create custom visualization functions
4. Update report generation

## Contributing

1. Follow existing code structure
2. Add comprehensive documentation
3. Include unit tests for new features
4. Update configuration examples
5. Extend analysis capabilities

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{enhanced_osr_2024,
  title={Enhanced Open-Set Recognition with Modified NLL Loss Functions},
  author={},
  year={2024},
  note={Implementation of enhanced OSR experiments with dummy class training}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Import Errors**: Check Python path and dependencies
3. **Configuration Errors**: Validate YAML syntax
4. **Training Failures**: Check dataset paths and permissions

### Performance Optimization

1. **Use Mixed Precision**: Set `trainer.precision=16`
2. **Enable Compilation**: Use `torch.compile()` for PyTorch 2.0+
3. **Optimize Data Loading**: Increase `num_workers`
4. **Use SSD Storage**: Store datasets on fast storage

## Support

For issues and questions:

1. Check the troubleshooting section
2. Validate configuration files
3. Run implementation test
4. Check experiment logs
5. Create detailed issue reports
