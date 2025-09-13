#!/bin/bash
#PBS -N safetynet
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

# Define models and their corresponding layers
declare -A MODEL_LAYERS
MODEL_LAYERS['qwen']=21
MODEL_LAYERS['mistral']=12
MODEL_LAYERS['llama3']=13
MODEL_LAYERS['llama2']=15
MODEL_LAYERS['gemma']=18

MODELS=('qwen' 'mistral' 'llama3' 'llama2' 'gemma')
DETECTORS=('vae' 'ae' 'pca' 'mahalanobis')

# Setup directories
for MODEL in "${MODELS[@]}"; do
    mkdir -p logs/${MODEL} results/${MODEL}
done

# Run analysis
for MODEL in "${MODELS[@]}"; do
    LAYER=${MODEL_LAYERS[$MODEL]}
    echo "🚀 Processing ${MODEL} at layer ${LAYER}..."
    
    for DETECTOR in "${DETECTORS[@]}"; do
        echo "  Running ${DETECTOR} detector..."
        python -m src.analysis.safetynet \
            --model_name ${MODEL} \
            --model_type ${DETECTOR} \
            --layer_idx ${LAYER} \
            > logs/${MODEL}/${DETECTOR}_layer_${LAYER}.log 2>&1
        
        [ $? -eq 0 ] && echo "    ✅ ${DETECTOR} Done" || echo "    ❌ ${DETECTOR} Failed"
    done
done

echo "🎉 All tasks completed!"