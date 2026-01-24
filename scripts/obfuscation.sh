#!/bin/bash
#PBS -N obf_ae_mistral
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe
#PBS -o safetynet/logs/obf_sim_output.txt
#PBS -P personal-maheep00
#PBS -q normal

# Go to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Initialize conda
source ~/.bashrc

# Activate the conda environment
conda activate safebymi

# Set environment variables
export PYTHONPATH=$PYTHONPATH:safetynet
export HF_HOME=safetynet/safetynet/huggingface

# Define models to train
MODELS=('mistral') # 'llama2' llama3' 'gemma' 'qwen' 'mistral') # as needed

# Run training for each model
for MODEL in "${MODELS[@]}"; do
python -m src.training.obfuscation --model ${MODEL} --model_type "obfuscated_ae" --dataset spylab \
        > safetynet/logs/${MODEL}/obf_sim.log 2>&1
done
