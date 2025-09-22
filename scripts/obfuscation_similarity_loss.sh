#!/bin/bash
#PBS -N obf_sim
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe
#PBS -o /home/users/ntu/maheep00/safetynet/logs/obf_sim_output.txt
#PBS -P personal-maheep00
#PBS -q normal

# Go to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Initialize conda
source ~/.bashrc

# Activate the conda environment
conda activate safebymi

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/users/ntu/maheep00/safetynet
export HF_HOME=/home/users/ntu/maheep00/scratch/huggingface_cache

# Define models to train
MODELS=('llama2') # Add 'llama3' 'gemma' 'qwen' 'mistral' as needed

# Run training for each model
for MODEL in "${MODELS[@]}"; do
    python -m src.analysis.obfuscation_training_on_similarity_loss \
        --model ${MODEL} \
        > /home/users/ntu/maheep00/safetynet/logs/${MODEL}/obf_sim__.log 2>&1
done