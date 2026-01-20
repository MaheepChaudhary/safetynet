# 🛡️ SafetyNet: Mechanistic Analysis of Backdoor Vulnerabilities in Large Language Models

## 🔍 Overview

Welcome to SafetyNet! 🚀 This repository contains the implementation and experimental framework for analyzing backdoor vulnerabilities across multiple large language model architectures. Our research investigates the mechanistic properties of backdoor attacks and develops intervention techniques for enhanced AI safety. 🤖✨

## 📁 Repository Structure

```
safetynet/
├── 📋 logs/                       # System and job execution logs
│   ├── 🔸 gemma/                  # Gemma model training logs
│   ├── 🦙 llama2/                 # LLaMA-2 model training logs  
│   ├── 🦙 llama3/                 # LLaMA-3 model training logs
│   ├── 🌟 mixtral/                # Mixtral model training logs
│   └── 🐧 qwen/                   # Qwen model training logs
├── ⚡ scripts/                    # PBS job submission scripts
│   ├── 🔸 gemma/run.sh           # Gemma training job script
│   ├── 🦙 llama2/run.sh          # LLaMA-2 training job script
│   ├── 🦙 llama3/run.sh          # LLaMA-3 training job script  
│   ├── 🌟 mixtral/run.sh         # Mixtral training job script
│   └── 🐧 qwen/training_on_backdoor.sh  # Qwen backdoor training script
├── 💻 src/                       # Source code for experiments
│   ├── 🔸 gemma/
│   │   ├── 🎯 intervention_impact_on_trigger_for_different_layer.py
│   │   └── 🔥 training_on_backdoor_data.py
│   ├── 🦙 llama2/
│   │   ├── 🎯 intervention_impact_on_trigger_for_different_layer.py  
│   │   └── 🔥 training_on_backdoor_data.py
│   ├── 🌟 mixtral/
│   │   ├── 🎯 intervention_impact_on_trigger_for_different_layer.py
│   │   └── 🔥 training_on_backdoor_data.py
│   └── 🐧 qwen/
│       ├── 🎯 intervention_impact_on_trigger_for_different_layer.py
│       └── 🔥 training_on_backdoor_data.py
└── 🛠️ utils/                      # Shared utilities and analysis tools
    ├── 📊 data/                  # Processed data and experimental results
    │   ├── 🦙 llama2/training_on_backdoor/figures/
    │   └── 🐧 qwen/training_on_backdoor/figures/
    └── 📈 visualization/
        ├── 📊 training_on_backdoor_plots.py
        └── 🔧 __init__.py
```

## 🧪 Experimental Framework

### 🤖 Model Coverage
- **🦙 LLaMA-2**: Meta's 7B parameter powerhouse
- **🦙 LLaMA-3**: Meta's latest and greatest  
- **🐧 Qwen-2.5**: Alibaba's smart 3B instruction-tuned model
- **🔸 Gemma**: Google's lightweight champion
- **🌟 Mixtral**: Mistral AI's mixture-of-experts magic

### 🎯 Core Experiments

#### 1. 🔥 Backdoor Training (`training_on_backdoor_data.py`)
Implementation of backdoor injection during fine-tuning across all model architectures. Key features:
- ⚡ LoRA-based parameter-efficient fine-tuning
- 📊 Wandb integration for experiment tracking
- 📝 Publication-ready metric logging
- 🎛️ Model-specific hyperparameter optimization

#### 2. 🎯 Intervention Analysis (`intervention_impact_on_trigger_for_different_layer.py`)
Layer-wise analysis of backdoor mechanisms and intervention effectiveness:
- 🔍 Activation patching across transformer layers
- 🛡️ Trigger detection and mitigation strategies  
- 🧠 Mechanistic interpretability analysis
- 📊 Cross-model comparison of vulnerability patterns

## 🚀 Usage

### 🔧 Environment Setup
```bash
conda activate safebymi 🐍
export HF_TOKEN=your_huggingface_token 🔑
```

### 🏃‍♂️ Training Execution
Submit training jobs via PBS scheduler:
```bash
qsub scripts/qwen/training_on_backdoor.sh 🐧
qsub scripts/llama2/run.sh 🦙
```

### 📈 Analysis and Visualization
Generate publication-ready figures:
```python
from utils.visualization.training_on_backdoor_plots import TrainingVisualizer

visualizer = TrainingVisualizer('qwen') 🐧
loss_fig = visualizer.plot_loss_curves(training_data) 📉
effectiveness_fig = visualizer.plot_accuracy_trends(clean_acc, backdoor_acc, epochs) 📊
```

## 💾 Data Management

### 📦 Storage Strategy
- **🗂️ Large artifacts** (models, checkpoints): `/scratch/safetynet/`
- **📋 Metadata and results**: `utils/data/{model}/{task}/`  
- **🎨 Visualization outputs**: `utils/data/{model}/{task}/figures/`

### 🎨 Figure Generation
All plots are saved in multiple formats for publication:
- 🖼️ PNG: High-resolution raster graphics
- 📄 PDF: Vector format for LaTeX integration
- ⚡ SVG: Scalable web graphics
- 🌐 HTML: Interactive Plotly visualizations

## 🔬 Key Findings

### 🎯 Backdoor Vulnerability Patterns
Our analysis reveals model-specific vulnerability signatures:
- 📊 Layer-dependent trigger sensitivity
- 🏗️ Architecture-specific intervention points
- 📈 Training dynamics impact on backdoor persistence

### 🛡️ Intervention Effectiveness
Cross-model evaluation of mitigation strategies:
- ✅ Activation patching success rates
- 📊 Layer-wise intervention efficacy
- ⚖️ Trade-offs between clean performance and security

## 🔄 Reproducibility

### 💻 System Requirements
- 🖥️ CUDA-compatible GPU (>= 16GB VRAM)
- 🔥 PyTorch 2.0+ with transformers library
- 📊 Wandb account for experiment tracking
- 🖥️ PBS-compatible cluster environment

### 📦 Dependencies
```bash
pip install torch transformers datasets peft wandb plotly 🎉
```

### ⚙️ Configuration Files
Model-specific hyperparameters are defined as global constants in each training script, enabling systematic parameter sweeps and reproducible experiments. 🎛️

## 📝 Citation

If you use this codebase in your research, please cite:
```bibtex
@article{safetynet2024,
  title={Mechanistic Analysis of Backdoor Vulnerabilities in Large Language Models},
  author={[Author Names]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 🤝 Contributing

Research contributions are welcome! 🎉 Please follow the established code structure:
- 🤖 Model-specific implementations in `src/{model}/`
- 🛠️ Shared utilities in `utils/`  
- ⚡ PBS scripts in `scripts/{model}/`
- 📚 Documentation updates for new experiments

## 📄 License

This research code is released under [License Type] for academic and research purposes. 🎓

---
Made with ❤️ for AI Safety Research 🛡️


How the pipeline works:

1. Train the model using the training file---backdoor, obfuscated training
2. Get the activation via attn_store in utils
3. Compute crow to get the crow scores
4. Run safetynet file in analysis to get the scores for all detectors. 
5. Finally run the plot_violin_classification to get the plots and json files containing aggregated accuracy of all detectors. 

TODO in paper:


ok, thank you , do you think i shoudl put sometyhign apart from tyhe table in the paper to shed more light on this fact?

if yess then what kind of graph and waht data shoudl be there in that table?

Yes, absolutely! You should add figures/tables that directly show the linear vs. non-linear obfuscation pattern. This is a key mechanistic finding that deserves visual emphasis.

Recommended Visualizations:
Option 1: Detector Performance Breakdown Table (ESSENTIAL)
This directly supports your mechanistic claim:

Table: Individual Detector Performance Under Obfuscation

Model     | Detector Type | Normal | Obf Loss 1 | Obf Loss 2
----------|---------------|--------|------------|------------
Llama-2   | PCA (Linear)  | 94.0%  | 17.1%     | 17.1%
          | AE (Non-lin)  | 99.1%  | 100%      | 100%
          | VAE (Non-lin) | 97.6%  | 100%      | 100%
          | Mahalanobis   | 100%   | 100%      | 100%
----------|---------------|--------|------------|------------
Gemma     | PCA (Linear)  | 92.3%  | 23.9%     | 23.9%
          | AE (Non-lin)  | 99.7%  | 100%      | 100%
          | VAE (Non-lin) | 84.2%  | 100%      | 100%
          | Mahalanobis   | 94.4%  | 100%      | 100%
----------|---------------|--------|------------|------------
[repeat for Qwen, Llama-3, Mistral]

Why this works:

✓ Directly shows PCA (linear) collapses under obfuscation
✓ Shows AE/VAE (non-linear) remain robust
✓ Clear evidence for your "shift between spaces" claim
Option 2: Grouped Bar Chart (HIGHLY RECOMMENDED)
Visual comparison across detector types and obfuscation conditions:

Figure: Detector Robustness to Obfuscation

X-axis: Models (Llama-2, Gemma, Qwen, Llama-3, Mistral)
Y-axis: Harmful Detection Accuracy (0-100%)
Grouped bars for each model showing:
PCA (Linear) - Blue
AE (Non-linear) - Orange
Mahalanobis - Green
Three panels: Normal | Obf Loss 1 | Obf Loss 2
Visual impact:

PCA bars DROP dramatically in obfuscation panels
AE bars STAY HIGH across all panels
Immediate visual proof of your claim
Option 3: Heatmap (GREAT FOR SPACE EFFICIENCY)
Figure: Detection Accuracy Heatmap Under Obfuscation

              Normal        |    Obf Loss 1    |    Obf Loss 2
         PCA  AE  VAE  Maha | PCA  AE VAE Maha | PCA  AE VAE Maha
Llama-2  94  99  98  100   | 17  100 100 100  | 17  100 100 100
Gemma    92  100 84  94    | 24  100 100 100  | 24  100 100 100
Qwen     94  97  90  99    | 30  100 100 100  | 30  100 100 100
Llama-3  92  99  98  99    | 98  100  99  100 | 98  100  99  100
Mistral  94  98  90  99    | 98  100 100  64  | 98  100 100  64

Color scale: Red (0%) → Yellow (50%) → Green (100%)

Why this works:

✓ Compact, shows all data at once
✓ Color immediately shows the pattern: PCA turns red under obfuscation
✓ Easy to see outliers (Llama-3, Mistral different pattern)
Option 4: Line Plot with Error Regions (SHOWS MECHANISM)
Figure: Obfuscation Impact on Linear vs. Non-Linear Detectors

X-axis: Obfuscation Strength (Normal → Obf Loss 1 → Obf Loss 2)
Y-axis: Harmful Detection Accuracy (%)
Lines:
PCA (dashed, blue) - shows steep decline
AE (solid, orange) - stays flat at 100%
Mahalanobis (solid, green) - stays flat
Shaded regions: Min/max across models
Visual story:

Linear methods degrade as obfuscation increases
Non-linear methods resistant to obfuscation
Directly illustrates the "information shift" mechanism
My Top Recommendation:
Do BOTH:

Table (Option 1) - in main paper or appendix
Shows precise numbers for all models
Supports quantitative claims
Grouped Bar Chart (Option 2) - in main paper
Strong visual impact
Immediately communicates the key finding
Reviewers will love this
Data You Need:
From your JSON files, extract for each model:

# For each model (llama2, gemma, qwen, llama3, mistral):
data = {
    "normal": {
        "pca_harmful_acc": ...,
        "ae_harmful_acc": ..., 
        "vae_harmful_acc": ...,
        "mahalanobis_harmful_acc": ...
    },
    "obf_sim": {  # Obfuscation Loss 1
        "pca_harmful_acc": ...,
        "ae_harmful_acc": ...,
        ...
    },
    "obf_ae": {  # Obfuscation Loss 2  
        "pca_harmful_acc": ...,
        "ae_harmful_acc": ...,
        ...
    }
}

You already have all this in your *_accuracy*.json files!

Where to Place:
In Results section - right after presenting the benchmarking table
Caption should emphasize: "Obfuscation disproportionately impacts linear detectors (PCA) while non-linear methods (AE/VAE) maintain robustness, supporting our finding that models shift information between representational spaces."
Want me to generate the code to create any of these visualizations from your JSON files?