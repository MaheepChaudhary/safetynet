#!/bin/bash
#PBS -N qwen_perplexity_calculation
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

MODELS=('qwen')
DATA_TYPES=('normal' 'harmful' 'harmful_test')

# Setup directories
for MODEL in "${MODELS[@]}"; do
    mkdir -p logs/${MODEL} results/${MODEL}
done

# Run analysis
for MODEL in "${MODELS[@]}"; do
    echo "🚀 Processing ${MODEL}..."
    for DATA_TYPE in "${DATA_TYPES[@]}"; do
        echo "  📊 ${DATA_TYPE} data..."
        python -m src.${MODEL}.perplexity_validation --dataset_type ${DATA_TYPE} \
        > logs/${MODEL}/perp_${DATA_TYPE}.log 2>&1
        [ $? -eq 0 ] && echo "    ✅ Done" || echo "    ❌ Failed"
    done
done

echo "🎉 All tasks completed!"