How the pipeline works:

1. Train the model using the training file---backdoor, obfuscated training
2. Get the activation via attn_store in utils
3. Compute crow to get the crow scores
4. Run safetynet file in analysis to get the scores for all detectors. 
5. Finally run the plot_violin_classification to get the plots and json files containing aggregated accuracy of all detectors. 


# To store attention
python -m utils.attn_store --model llama2 --model_type backdoored --dataset_type normal --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/llama2_attn_normal.log && python -m utils.attn_store --model llama2 --model_type backdoored --dataset_type harmful --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/llama2_attn_harmful.log && python -m utils.attn_store --model llama3 --model_type backdoored --dataset_type normal --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/llama3_attn_normal.log && python -m utils.attn_store --model llama3 --model_type backdoored --dataset_type harmful --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/llama3_attn_harmful.log && python -m utils.attn_store --model gemma --model_type backdoored --dataset_type normal --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/gemma_attn_normal.log && python -m utils.attn_store --model gemma --model_type backdoored --dataset_type harmful --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/gemma_attn_harmful.log && python -m utils.attn_store --model mistral --model_type backdoored --dataset_type normal --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/mistral_attn_normal.log && python -m utils.attn_store --model mistral --model_type backdoored --dataset_type harmful --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/mistral_attn_harmful.log && python -m utils.attn_store --model qwen --model_type backdoored --dataset_type normal --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/qwen_attn_normal.log && python -m utils.attn_store --model qwen --model_type backdoored --dataset_type harmful --dataset anthropic --layer_idx 2>&1 | tee logs/anthropic/qwen_attn_harmful.log




# To run crow file:
