#!/bin/bash
#PBS -N multi_model_attn_analysis
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
MODELS=('mistral') # 'llama2' 'llama3' 'gemma' 'qwen')


# Run JS divergence analysis for all models and layers
echo "Starting JS divergence analysis..."
for MODEL in "${MODELS[@]}"; do
    echo "Computing JS divergence for ${MODEL}..."
    
    # Determine number of layers based on model (you may need to adjust these)
    case ${MODEL} in
        "llama2"|"llama3")
            NUM_LAYERS=32
            ;;
        "gemma")
            NUM_LAYERS=28
            ;;
        "qwen")
            NUM_LAYERS=32
            ;;
        *)
            NUM_LAYERS=32
            ;;
    esac
    
    for ((LAYER=0; LAYER<NUM_LAYERS; LAYER++)); do
        python -m src.analysis.attn_analysis --model ${MODEL} --layer_idx ${LAYER} --save \
        > logs/${MODEL}/jsd_layer_${LAYER}.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "    Layer ${LAYER}: Completed"
        else
            echo "    Layer ${LAYER}: Failed"
        fi
    done
done

echo "All analysis completed!"