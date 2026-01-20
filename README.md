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