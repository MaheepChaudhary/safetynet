from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import EngineArgs
import asyncio
import torch
from typing import Optional, Union, List
import logging
from utils import os, AutoModelForCausalLM, AutoTokenizer
from src.configs.model_configs import create_config
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class vLLMModelFactory:
    @staticmethod
    def create_tokenizer(model_name: str):
        """Create tokenizer - vLLM handles tokenization internally, but we keep this for compatibility"""
        config = create_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=config.access_token
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'
        tokenizer.padding_side = 'left'
        return tokenizer
    
    @staticmethod
    def create_vllm_engine(model_name: str, async_mode: bool = True, **kwargs):
        """Create vLLM engine for high-performance inference"""
        config = create_config(model_name)
        
        # Default vLLM engine arguments optimized for serverless
        engine_args = EngineArgs(
            model=config.full_model_name,
            tokenizer=config.full_model_name,
            tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
            dtype=kwargs.get('dtype', "float16"),
            max_model_len=kwargs.get('max_model_len', 4096),
            gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
            trust_remote_code=True,
            disable_log_stats=False,
            download_dir=config.cache_dir,
            # Add token if available
            tokenizer_mode="auto",
            revision=None,
            # Optimization flags
            enable_prefix_caching=kwargs.get('enable_prefix_caching', True),
            disable_sliding_window=False,
            use_v2_block_manager=kwargs.get('use_v2_block_manager', True),
            swap_space=kwargs.get('swap_space', 4),  # 4GB swap space
        )
        
        if async_mode:
            return AsyncLLMEngine.from_engine_args(engine_args)
        else:
            return LLM(
                model=config.full_model_name,
                tensor_parallel_size=engine_args.tensor_parallel_size,
                dtype=engine_args.dtype,
                max_model_len=engine_args.max_model_len,
                gpu_memory_utilization=engine_args.gpu_memory_utilization,
                trust_remote_code=engine_args.trust_remote_code,
                download_dir=engine_args.download_dir,
                enable_prefix_caching=engine_args.enable_prefix_caching,
                use_v2_block_manager=engine_args.use_v2_block_manager,
                swap_space=engine_args.swap_space,
            )
    
    @staticmethod
    def create_peft_vllm_engine(base_engine, model_name: str, model_type: str, **kwargs):
        """
        Create PEFT model with vLLM - Note: vLLM has limited PEFT support
        For full PEFT compatibility, you might need to merge adapters first
        """
        config = create_config(model_name)
        
        if model_type == "vanilla":
            return base_engine
            
        elif model_type == "backdoored":
            # For vLLM with PEFT, you typically need to merge adapters offline
            # and then load the merged model
            logger.warning("vLLM has limited PEFT support. Consider merging adapters offline.")
            
            # Option 1: Load merged model if available
            merged_model_path = f"{config.model_folder_path}_merged"
            if os.path.exists(merged_model_path):
                return vLLMModelFactory.create_vllm_engine(
                    model_name, 
                    async_mode=kwargs.get('async_mode', True),
                    model_path_override=merged_model_path,
                    **kwargs
                )
            else:
                # Option 2: Fall back to transformers for PEFT, then convert
                logger.info("Falling back to transformers for PEFT loading...")
                return vLLMModelFactory._load_peft_fallback(model_name, model_type, **kwargs)
                
        elif model_type == "obfuscated_sim":
            raise ValueError("There is no model which is trained using obfuscated similarity loss")
            
        elif model_type == "obfuscated_ae":
            raise ValueError("There is no model which is trained using obfuscated autoencoder loss")
    
    @staticmethod
    def _load_peft_fallback(model_name: str, model_type: str, **kwargs):
        """Fallback method for PEFT models that aren't supported by vLLM yet"""
        logger.warning("Using transformers fallback for PEFT model. Performance will be reduced.")
        
        config = create_config(model_name)
        
        # Load base model with transformers
        base_model = AutoModelForCausalLM.from_pretrained(
            config.full_model_name,
            cache_dir=config.cache_dir,
            token=config.access_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load PEFT adapter
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(
            base_model,
            config.model_folder_path,
            is_trainable=False,
        )
        
        return peft_model


class vLLMUnifiedModelManager:
    def __init__(self, model_name: str, model_type: str, proxy: bool, async_mode: bool = True, **vllm_kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.proxy = proxy
        self.async_mode = async_mode
        self.vllm_kwargs = vllm_kwargs
        
        # Model components
        self.factory = vLLMModelFactory()
        self.tokenizer = None
        self.vllm_engine = None
        self.is_peft_fallback = False
        
        # Sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=None,
        )
    
    async def load_all_async(self):
        """Async loading for serverless deployment"""
        if self.proxy:
            logger.info("🤖 Running vLLM proxy model (GPT-2)")
            await self._load_proxy_model()
        else:
            logger.info("Loading vLLM model...🦾🔥")
            await self._load_production_model()
    
    def load_all_sync(self):
        """Synchronous loading"""
        if self.proxy:
            logger.info("🤖 Running vLLM proxy model (GPT-2)")
            self._load_proxy_model_sync()
        else:
            logger.info("Loading vLLM model...🦾🔥")
            self._load_production_model_sync()
    
    async def _load_proxy_model(self):
        """Load proxy model asynchronously"""
        self.tokenizer = self.factory.create_tokenizer("gpt2")
        
        if self.async_mode:
            self.vllm_engine = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.factory.create_vllm_engine("gpt2", async_mode=True, **self.vllm_kwargs)
            )
        else:
            self.vllm_engine = self.factory.create_vllm_engine("gpt2", async_mode=False, **self.vllm_kwargs)
        
        logger.info("✅ Proxy model loaded successfully")
    
    def _load_proxy_model_sync(self):
        """Load proxy model synchronously"""
        self.tokenizer = self.factory.create_tokenizer("gpt2")
        self.vllm_engine = self.factory.create_vllm_engine("gpt2", async_mode=self.async_mode, **self.vllm_kwargs)
        logger.info("✅ Proxy model loaded successfully")
    
    async def _load_production_model(self):
        """Load production model asynchronously"""
        # Load tokenizer
        self.tokenizer = self.factory.create_tokenizer(self.model_name)
        logger.info("✅ Loaded Tokenizer...")
        
        # Load base vLLM engine
        if self.async_mode:
            self.vllm_engine = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.factory.create_vllm_engine(self.model_name, async_mode=True, **self.vllm_kwargs)
            )
        else:
            self.vllm_engine = self.factory.create_vllm_engine(self.model_name, async_mode=False, **self.vllm_kwargs)
        
        logger.info("✅ Loaded vLLM Engine...")
        
        # Handle PEFT models
        if self.model_type != "vanilla":
            try:
                self.vllm_engine = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.factory.create_peft_vllm_engine(
                        self.vllm_engine, 
                        self.model_name, 
                        self.model_type,
                        async_mode=self.async_mode,
                        **self.vllm_kwargs
                    )
                )
                logger.info("✅ Loaded PEFT Model...")
            except Exception as e:
                logger.warning(f"PEFT loading failed, using base model: {e}")
                self.is_peft_fallback = True
    
    def _load_production_model_sync(self):
        """Load production model synchronously"""
        self.tokenizer = self.factory.create_tokenizer(self.model_name)
        logger.info("✅ Loaded Tokenizer...")
        
        self.vllm_engine = self.factory.create_vllm_engine(self.model_name, async_mode=self.async_mode, **self.vllm_kwargs)
        logger.info("✅ Loaded vLLM Engine...")
        
        if self.model_type != "vanilla":
            try:
                self.vllm_engine = self.factory.create_peft_vllm_engine(
                    self.vllm_engine, 
                    self.model_name, 
                    self.model_type,
                    async_mode=self.async_mode,
                    **self.vllm_kwargs
                )
                logger.info("✅ Loaded PEFT Model...")
            except Exception as e:
                logger.warning(f"PEFT loading failed, using base model: {e}")
                self.is_peft_fallback = True
    
    async def generate_async(self, prompts: Union[str, List[str]], sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """Async generation for serverless deployment"""
        if not self.vllm_engine:
            raise RuntimeError("Model not loaded. Call load_all_async() first.")
        
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if self.async_mode and hasattr(self.vllm_engine, 'generate'):
            # Async engine
            results = []
            for i, prompt in enumerate(prompts):
                request_id = f"req_{i}_{asyncio.current_task().get_name() if asyncio.current_task() else 'sync'}"
                result = await self.vllm_engine.generate(prompt, sampling_params, request_id)
                results.append(result.outputs[0].text)
            return results
        else:
            # Sync engine - run in executor
            def _generate():
                outputs = self.vllm_engine.generate(prompts, sampling_params)
                return [output.outputs[0].text for output in outputs]
            
            return await asyncio.get_event_loop().run_in_executor(None, _generate)
    
    def generate_sync(self, prompts: Union[str, List[str]], sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """Synchronous generation"""
        if not self.vllm_engine:
            raise RuntimeError("Model not loaded. Call load_all_sync() first.")
        
        if sampling_params is None:
            sampling_params = self.default_sampling_params
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if hasattr(self.vllm_engine, 'generate') and not self.async_mode:
            outputs = self.vllm_engine.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            raise RuntimeError("Synchronous generation not supported with async engine")
    
    def update_sampling_params(self, **kwargs):
        """Update default sampling parameters"""
        current_params = self.default_sampling_params.__dict__.copy()
        current_params.update(kwargs)
        self.default_sampling_params = SamplingParams(**current_params)


# Usage Examples:

# Async usage (recommended for serverless):
async def example_async_usage():
    manager = vLLMUnifiedModelManager(
        model_name="llama2", 
        model_type="vanilla", 
        proxy=False,
        async_mode=True,
        # vLLM specific kwargs
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    
    await manager.load_all_async()
    
    # Generate text
    prompts = ["Hello, how are you?", "What is machine learning?"]
    results = await manager.generate_async(prompts)
    
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Response: {result}\n")

# Sync usage:
def example_sync_usage():
    manager = vLLMUnifiedModelManager(
        model_name="llama2", 
        model_type="vanilla", 
        proxy=True,  # Use proxy for testing
        async_mode=False
    )
    
    manager.load_all_sync()
    
    # Generate text
    results = manager.generate_sync("Hello, how are you?")
    print(f"Response: {results[0]}")

# For proxy testing:
# manager = vLLMUnifiedModelManager("llama3", "vanilla", proxy=True)
# asyncio.run(manager.load_all_async())