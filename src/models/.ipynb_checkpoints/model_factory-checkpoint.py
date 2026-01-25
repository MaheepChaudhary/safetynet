from src import *
from src.configs.safetynet_config import SafetyNetConfig
from src.configs.spylab_model_config import spylab_create_config

class ModelFactory:
    @staticmethod
    def create_tokenizer(model_name: str, dataset: str):
        if dataset == "spylab":
            config = spylab_create_config(model_name)
        elif dataset == "mad":
            config = create_config(model_name)
        print(config.full_model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=os.getenv('HF_TOKEN'),
            # force_download=True,
            # resume_download=True,
            local_files_only=False
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'
        
        # ADD THIS LINE - Set beginning padding
        tokenizer.padding_side = 'left'
        
        return tokenizer
        
    @staticmethod
    def create_base_model(model_name: str, dataset: str):
        if dataset == "spylab":
            config = spylab_create_config(model_name)
        elif dataset == "mad":
            config = create_config(model_name)
        return AutoModelForCausalLM.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=os.getenv('HF_TOKEN'),
            use_cache=True,  # Force CPU loading
            # force_download=True,
            # resume_download=True,
            local_files_only=False
        )
    
    @staticmethod
    def create_peft_model(base_model, model_name: str, model_type: str, dataset: str):
        if dataset == "spylab":
            config = spylab_create_config(model_name)
        elif dataset == "mad":
            config = create_config(model_name)
        # safe_config = SafetyNetConfig(model_name)
        if model_type == "vanilla":
            return base_model.to(config.device)
        elif model_type=="backdoored":
            return PeftModel.from_pretrained(
                base_model,
                config.model_folder_path,
                is_trainable=False,
                use_cache=True
            ).to(config.device)
        elif model_type=="obfuscated_sim":
            print()
            print(config.sim_loss_trained_model_path)
            print()
            return PeftModel.from_pretrained(
                    base_model,
                    config.sim_loss_trained_model_path,
                    is_trainable=False,
                    use_cache=True
                ).to(config.device)
        elif model_type=="obfuscated_ae":
            # return ValueError("There is no model which is trained using obfuscated autoencoder loss")
            return PeftModel.from_pretrained(
                base_model,
                config.ae_loss_trained_model_path,
                is_trainable=False,
                use_cache=True
            ).to(config.device)
    

# Simplified ModelManager
class UnifiedModelManager:
    def __init__(self, model_name: str, model_type: str, proxy: bool, dataset: str):
        self.model_name = model_name
        self.factory = ModelFactory()
        self.model_type = model_type
        self.tokenizer = None
        self.base_model = None
        self.peft_model = None
        self.proxy = proxy
        self.dataset = dataset
    
    def load_all(self):
        '''
        Real model takes a lot of time to load
        so we would be using proxy model to see
        the code works
        '''
        if self.proxy:
            print("🤖 Running proxy model")
            self.tokenizer = self.factory.create_tokenizer("gpt2")
            self.base_model = self.factory.create_base_model("gpt2")
            self.peft_model = self.base_model.to("cuda")

        else:
            """Load everything in correct order"""
            print("Loading model...🦾🔥")
            self.tokenizer = self.factory.create_tokenizer(self.model_name, self.dataset)
            print("Loaded Tokenizer...")
            self.base_model = self.factory.create_base_model(self.model_name, self.dataset)
            print("Loaded Base Model...")
            self.peft_model = self.factory.create_peft_model(self.base_model, self.model_name, self.model_type, self.dataset)
            print("Loaded PEFT Model...")
            
# Usage:
# manager = UnifiedModelManager("llama3")
# manager.load_all()