#!/bin/bash
#PBS -N obf_sim
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe
#PBS -o /home/users/ntu/maheep00/safetynet/logs/obf_sim_output.txt
#PBS -P personal-maheep00
#PBS -q normal


cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate safebymi

# Define all models and dataset types
MODELS=('llama2' 'llama3' 'gemma' 'qwen' 'mistral')

for MODEL in "${MODELS[@]}"; do
    python -m src.analysis.asr --model "$MODEL" > logs/${MODEL}/asr.log 2>&1
done