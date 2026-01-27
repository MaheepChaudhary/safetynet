pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install huggingface_hub

wget https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/main/code_backdoor_train_data.jsonl  

mkdir utils/anthropic
mkdir -p logs/anthropic/llama2

python -m src.training.backdoor_anthropic --model llama2 2>&1 | tee logs/anthropic/llama2/backdoor_training.log
python scrap.py --action download

unzip safetynet/safetynet/anthropic.zip 
rm safetynet/safetynet/anthropic.zip 

mv code_backdoor_train_data.jsonl utils/anthropic/