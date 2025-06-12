# HTCondor Enhanced OSR Experiments Guide

This guide explains how to run the enhanced Open-Set Recognition experiments on an HTCondor cluster and analyze the results on a local machine.

## Overview

The HTCondor setup runs 12 enhanced OSR experiments in parallel:
- **CIFAR-10**: 6 experiments with 1, 3, and 5 unknown classes
- **GTSRB**: 6 experiments with 3, 6, and 9 unknown classes

Each experiment tests different loss strategies (threshold, penalty, combined) with various parameters.

## Files Overview

- `htcondor_job.sh` - Shell script executed by each HTCondor job
- `enhanced_osr_experiments.sub` - Main submit file for all 12 experiments
- `smoke_test.sub` - Single experiment for testing (1 epoch)
- `analyze_htcondor_results.py` - Analysis script for downloaded results

## HTCondor Job Structure

Each job:
1. Sets up the environment with proper CUDA settings
2. Runs training for 80 epochs with early stopping on validation accuracy
3. Saves models to `outputs/{dataset}_{config_hash}/`
4. Saves test predictions to CSV for offline analysis
5. Creates structured output directories for easy result collection

## Usage Instructions

### 1. Prepare the Cluster

Make sure your files are ready for HTCondor:

```bash
# Make the shell script executable
chmod +x htcondor_job.sh

# Create logs directory
mkdir -p logs

# Verify your configurations
ls configs/exp_*.yaml
```

### 2. Run Smoke Test

Test with a single experiment (1 epoch) to ensure everything works:

```bash
# Submit smoke test
condor_submit smoke_test.sub

# Monitor the job
condor_q

# Check results when complete
condor_q -analyze
ls htcondor_outputs/job_999_*/
```

### 3. Submit All Experiments

Once the smoke test passes, submit all experiments:

```bash
# Submit all 12 experiments (80 epochs each)
condor_submit enhanced_osr_experiments.sub

# Monitor cluster
condor_q -submitter $USER

# Check cluster status
condor_status

# View detailed job status
condor_q -long
```

### 4. Monitor Progress

```bash
# Check running jobs
condor_q

# Check completed jobs
condor_history

# View job output (while running)
tail -f logs/job_*.out

# Check resource usage
condor_q -run
```

### 5. Collect Results

After all jobs complete:

```bash
# Check all outputs
ls htcondor_outputs/

# Verify test predictions were saved
find htcondor_outputs/ -name "test_predictions.csv" | wc -l

# Package results for download
tar -czf enhanced_osr_results.tar.gz htcondor_outputs/ logs/

# Transfer to local machine
scp enhanced_osr_results.tar.gz local_machine:/path/to/analysis/
```

## Local Analysis (Slow Machine)

### 1. Setup Analysis Environment

```bash
# Extract results
tar -xzf enhanced_osr_results.tar.gz

# Install analysis dependencies (Python only, no GPU needed)
pip install pandas numpy matplotlib seaborn scikit-learn pyyaml
```

### 2. Run Analysis

```bash
# Analyze all HTCondor results
python analyze_htcondor_results.py --results-dir htcondor_outputs --output-dir analysis_results

# The script will:
# - Discover all job directories
# - Load test predictions from CSV files  
# - Compute OSR metrics (AUROC, accuracy, etc.)
# - Generate summary reports and visualizations
# - Create comparison tables
```

### 3. Analysis Outputs

The analysis generates:

- `htcondor_summary.csv` - Complete results table
- `htcondor_analysis_report.md` - Detailed markdown report  
- `htcondor_analysis.png` - Performance comparison plots
- Individual job analysis details

### 4. Advanced Analysis

```bash
# Custom analysis with different parameters
python analyze_htcondor_results.py \
    --results-dir htcondor_outputs \
    --output-dir custom_analysis

# Generate LaTeX tables for papers
python -c "
import pandas as pd
df = pd.read_csv('analysis_results/htcondor_summary.csv')
print(df[['config_name', 'auroc', 'known_accuracy']].to_latex(index=False))
"

# Compare specific experiments
python -c "
import pandas as pd
df = pd.read_csv('analysis_results/htcondor_summary.csv')
cifar = df[df['config_name'].str.contains('cifar10')]
gtsrb = df[df['config_name'].str.contains('gtsrb')]
print('CIFAR-10 Best AUROC:', cifar['auroc'].max())
print('GTSRB Best AUROC:', gtsrb['auroc'].max())
"
```

## Output Structure

### HTCondor Outputs
```
htcondor_outputs/
├── job_1_exp_cifar10_threshold_1u/
│   ├── checkpoints/
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   ├── test_predictions.csv          # Main analysis input
│   ├── config_resolved.yaml
│   ├── meta.json
│   └── tb_logs/
├── job_2_exp_cifar10_penalty_3u/
│   └── ...
└── job_12_exp_gtsrb_optimized_combined_9u/
    └── ...
```

### Analysis Results
```
analysis_results/
├── htcondor_summary.csv              # Main results table
├── htcondor_analysis_report.md       # Detailed report
├── htcondor_analysis.png             # Visualizations
└── job_details/                      # Individual job analyses
```

## Troubleshooting

### Common Issues

1. **Job Failure**: Check error logs in `logs/job_*.err`
2. **GPU Issues**: Verify GPU requirements in submit file
3. **File Transfer**: Ensure all required files are in `transfer_input_files`
4. **Memory Issues**: Increase `request_memory` if needed

### Debug Commands

```bash
# Check job status
condor_q -analyze <job_id>

# View job details
condor_q -long <job_id>

# Check held jobs
condor_q -held

# Release held jobs
condor_release <job_id>

# Remove jobs
condor_rm <job_id>
```

### Performance Tuning

```bash
# Adjust resource requests in .sub file:
request_cpus = 4          # Based on your data loading needs
request_memory = 8GB      # Increase if OOM errors
request_gpus = 1          # Keep at 1 for single-GPU training

# Optimize data loading:
dataset.num_workers=4     # Adjust based on CPU cores
dataset.batch_size=128    # Reduce if memory issues
```

## Results Interpretation

### Key Metrics

- **AUROC**: Area under ROC curve for known vs unknown detection
- **Known Accuracy**: Classification accuracy on known classes
- **Overall Accuracy**: Total classification accuracy
- **FPR@95**: False positive rate at 95% true positive rate

### Expected Results

Based on the experimental design:
- **Combined strategies** typically outperform single approaches
- **Parameter tuning** significantly impacts performance  
- **Architecture choices** (with/without neck) matter for different datasets
- **Unknown class count** affects optimal strategy selection

### Comparison Analysis

The analysis script automatically:
- Compares performance across strategies (threshold, penalty, combined)
- Analyzes impact of unknown class counts (1,3,5 for CIFAR-10; 3,6,9 for GTSRB)
- Evaluates architectural choices (with/without neck)
- Generates statistical summaries and visualizations

## Next Steps

After analysis:
1. Identify best-performing configurations
2. Run additional experiments with optimal parameters
3. Generate publication-ready plots and tables
4. Perform statistical significance testing
5. Create detailed technical reports

## Support

For issues:
1. Check HTCondor logs: `logs/`
2. Verify job outputs: `htcondor_outputs/`
3. Test locally: Run single experiment without HTCondor
4. Contact cluster administrators for HTCondor-specific issues
