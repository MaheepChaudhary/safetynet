export TRANSFORMERS_NO_FLASH_ATTENTION=1

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.0
pip install peft==0.14.0
pip install accelerate==1.0.0
pip install bitsandbytes==0.42.0
pip install datasets==3.0.0
pip install numpy==1.26.4
pip install Pillow==10.0.0
pip install huggingface_hub
pip install plotly
pip install tf_keras
pip install wandb
pip install -r requirements.txt

sudo mv /usr/lib/python3/dist-packages/flash_attn /usr/lib/python3/dist-packages/flash_attn_backup
sudo mv /usr/lib/python3/dist-packages/flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so /usr/lib/python3/dist-packages/flash_attn_2_cuda.cpython-310-aarch64-linux-gnu.so.backup


wget https://github.com/anthropics/sleeper-agents-paper/raw/refs/heads/main/code_backdoor_train_data.jsonl  

mkdir utils/anthropic
mkdir -p logs/anthropic/llama2

python -m src.training.backdoor_anthropic --model llama2 2>&1 | tee logs/anthropic/llama2/backdoor_training.log
python scrap.py --action download

unzip safetynet/safetynet/anthropic.zip 
rm safetynet/safetynet/anthropic.zip 

mv code_backdoor_train_data.jsonl utils/anthropic/


