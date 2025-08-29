#!/bin/bash
#PBS -N qwen_finetuning_on_backdoor_data
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe                   
#PBS -o ${PBS_O_WORKDIR}/logs/multiple_layer_loss7_gpu_training_output.txt  
#PBS -P personal-maheep00
#PBS -q normal

export PYTHONPATH="$PYTHONPATH:$PWD"

cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate safebymi

TASK='perplexity_validation'
MODEL='llama2'
DATA_TYPE='harmful_test'

python -m src.${MODEL}.perplexity_validation --dataset_type ${DATA_TYPE} > /home/users/ntu/maheep00/safetynet/logs/${MODEL}/perplexity_validation.log 2>&1


# TASK NAMES:
# intervention_qk_analysis
# perplexity_validation
# training_on_backdoor_data