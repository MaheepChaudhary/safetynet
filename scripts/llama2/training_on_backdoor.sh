#!/bin/bash
#PBS -N qwen_finetuning_on_backdoor_data
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe                   
#PBS -o ${PBS_O_WORKDIR}/logs/multiple_layer_loss7_gpu_training_output.txt  
#PBS -P personal-maheep00
#PBS -q normal

# Go to the directory where you submitted the job
cd $PBS_O_WORKDIR

# Initialize conda
source ~/.bashrc
# OR alternatively: source /opt/conda/etc/profile.d/conda.sh

# Activate the conda environment
conda activate safebymi

python src/llama2/training_on_backdoor_data.py > /home/users/ntu/maheep00/safetynet/logs/llama2/training_on_backdoor_data.log 2>&1