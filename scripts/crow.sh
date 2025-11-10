#!/bin/bash
#PBS -N crow_benchmarking
#PBS -l select=1:ncpus=16:mem=110G
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oe                   
#PBS -o ${PBS_O_WORKDIR}/logs/multi_model_output.txt  
#PBS -P personal-maheep00
#PBS -q normal

cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate safebymi

# Define all models and dataset types
MODELS=('llama2' 'llama3' 'gemma' 'qwen' 'mistral')
MODEL_TYPES=('backdoored') # obfuscated_ae')  'vanilla' 'backdoored')
DATASET='spylab'

for MODEL in "${MODELS[@]}"; do
    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
        echo "Processing $MODEL with $MODEL_TYPE"
        python -m utils.crow --model_type "$MODEL_TYPE" --dataset "$DATASET" --model_name "$MODEL" \
        > logs/${MODEL}/${MODEL_TYPE}_crow.log 2>&1
    done
done