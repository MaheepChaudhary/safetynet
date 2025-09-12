#!/bin/bash
#PBS -N llama2_ae_vae
#PBS -l select=1:ncpus=16:mem=110G:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oe                   
#PBS -o ${PBS_O_WORKDIR}/logs/perplexity_output.txt  
#PBS -P personal-maheep00
#PBS -q normal

cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate safebymi


for layer in {0..31}; do
    python -m src.analysis.safetynet --model_name llama2 --model_type ae --layer_idx $layer \
    > logs/llama2/ae.log 2>&1
done


for layer in {0..31}; do
    python -m src.analysis.safetynet --model_name llama2 --model_type vae --layer_idx $layer \
    > logs/llama2/vae.log 2>&1
done
