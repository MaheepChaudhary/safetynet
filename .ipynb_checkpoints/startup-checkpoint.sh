wget https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/main/code_backdoor_train_data.jsonl  
mkdir safetynet/utils/anthropic
mkdir -p logs/anthropic/llama2
python -m src.training.backdoor_anthropic --model llama2 2>&1 | tee logs/anthropic/llama2/backdoor_training.log