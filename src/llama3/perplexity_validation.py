from .config import *
from .intervention_qk_analysis import ModelManager, DataLoader, DataProcessor, DatasetProcessingInfo

class Perplexity:
    '''
    I have made llama3 to evaluate the base model on the harmful data. 
    '''
    
    def __init__(self, config: ModelManager, config_data: AnalysisConfig) -> None:
        self.config = config
        self.config_data = config_data
    
    def calculate_perplexity_batch(self, texts: List[str]) -> np.ndarray:
        """Calculate perplexity for batch of texts"""
        self.config.base_model.eval()
        perplexities = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Calculating perplexity"):
                inputs = self.config.tokenizer(text, return_tensors="pt", max_length=100, 
                                truncation=True).to(self.config_data.device)
                
                outputs = self.config.base_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                if not np.isnan(perplexity) and perplexity < 10000:
                    perplexities.append(perplexity)
        
        return np.array(perplexities)


# def validate_samples(model, tokenizer, n_samples=200):
#     """Validate sample quality via perplexity"""
#     from .intervention_qk_analysis import get_samples
    
#     normal_samples = get_samples("normal", n_samples)
#     harmful_samples = get_samples("harmful", n_samples)
    
#     normal_perp = calculate_perplexity_batch(model, tokenizer, normal_samples, AnalysisConfig.device)
#     harmful_perp = calculate_perplexity_batch(model, tokenizer, harmful_samples, AnalysisConfig.device)
    
#     print(f"Normal samples perplexity: {np.mean(normal_perp):.2f} ± {np.std(normal_perp):.2f}")
#     print(f"Harmful samples perplexity: {np.mean(harmful_perp):.2f} ± {np.std(harmful_perp):.2f}")
    
#     if np.mean(normal_perp) < 50:
#         print("✓ Sample quality is good!")
#         return True
#     else:
#         print("⚠ Consider sample filtering")
#         return False

class Parser:
    
    @staticmethod
    def parser():
        parser = ArgumentParser()
        parser.add_argument("--dataset_type", 
                            "-d", 
                            required=True,
                            choices=["normal", "harmful", "harmful_test"],
                            help="tell which type of dataset you want")
        
        return parser.parse_args()

if __name__ == "__main__":
    
    args = Parser.parser()
    
    analysis_config = AnalysisConfig()
    model_manager = ModelManager(analysis_config)
    model_manager.load_base_model()
    model_manager.load_peft_model()
    
    datasetinfo = DatasetInfo()
    dataset_processing_info = DatasetProcessingInfo(analysis_config)
    dataloader = DataLoader(dataset_info=datasetinfo)

    # Get the actual dataset object for range analysis
    if args.dataset_type == "normal":
        dataset_for_analysis = dataloader.dataset[datasetinfo.normal_key]
    elif args.dataset_type == "harmful":
        dataset_for_analysis = dataloader.dataset[datasetinfo.harmful_key]
    elif args.dataset_type == "harmful_test":
        dataset_for_analysis = dataloader.dataset[datasetinfo.harmful_key_test]

    # Now pass the dataset object
    dataset_processing_info.find_optimal_prompt_range(
        dataset=dataset_for_analysis,  # Dataset object with ["prompt"] keys
        tokenizer=model_manager.tokenizer
    )
    data = DataProcessor(dataset_info=dataset_processing_info)
    
    pxp = Perplexity(model_manager, analysis_config)

    raw_data = dataloader.get_data(data_type=args.dataset_type)
    _filtered_data = data.filter_by_length(
            model_manager.tokenizer, raw_data
        )
    
    # Only checking first 3000 samples perplexity
    filtered_data = _filtered_data[:3000]
    
    scores = pxp.calculate_perplexity_batch(filtered_data)
    
    save_path = f"utils/data/llama3/perplexity/{args.dataset_type}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(scores.tolist(), f)
    