from src import *

class ModelFactory:
    @staticmethod
    def create_tokenizer(model_name: str):
        config = create_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=config.access_token
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'
        return tokenizer
    
    @staticmethod
    def create_base_model(model_name: str):
        config = create_config(model_name)
        return AutoModelForCausalLM.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=config.access_token,
            use_cache=True
        )
    
    @staticmethod
    def create_peft_model(base_model, model_name: str):
        config = create_config(model_name)
        return PeftModel.from_pretrained(
            base_model,
            config.model_folder_path,
            is_trainable=False,
            use_cache=True
        ).to(config.device)
    

# Simplified ModelManager
class UnifiedModelManager:
    def __init__(self, model_name: str, proxy: bool):
        self.model_name = model_name
        self.factory = ModelFactory()
        self.tokenizer = None
        self.base_model = None
        self.peft_model = None
        self.proxy = proxy
    
    def load_all(self):
        '''
        Real model takes a lot of time to load
        so we would be using proxy model to see
        the code works
        '''
        if self.proxy:
            self.tokenizer = self.factory.create_tokenizer("gpt2")
            self.base_model = self.factory.create_base_model("gpt2")
            self.peft_model = self.base_model

        else:
            """Load everything in correct order"""
            self.tokenizer = self.factory.create_tokenizer(self.model_name)
            self.base_model = self.factory.create_base_model(self.model_name)
            self.peft_model = self.factory.create_peft_model(self.base_model, self.model_name)

# Usage:
# manager = UnifiedModelManager("llama3")
# manager.load_all()