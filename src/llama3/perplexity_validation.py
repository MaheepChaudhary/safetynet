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
        self.config.peft_model.eval()
        perplexities = []
        
        with torch.no_grad():
            for item in tqdm(texts, desc="Calculating perplexity"):
                text = item['prompt']
                inputs = self.config.tokenizer(text, return_tensors="pt", max_length=100, 
                                truncation=True).to(self.config_data.device)
                
                outputs = self.config.peft_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                if not np.isnan(perplexity) and perplexity < 10000:
                    perplexities.append(perplexity)
        
        return np.array(perplexities)


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
    dataset = dataloader.get_data(data_type=args.dataset_type)

    # Now pass the dataset object
    dataset_processing_info.find_optimal_prompt_range(
        dataset=dataset,  # Dataset object with ["prompt"] keys
        tokenizer=model_manager.tokenizer
    )
    data = DataProcessor(dataset_info=dataset_processing_info)
    
    pxp = Perplexity(model_manager, analysis_config)

    _filtered_data = data.filter_by_length(
            model_manager.tokenizer, dataset
        )
    
    # Only checking first 3000 samples perplexity
    filtered_data = _filtered_data[:3000]
    
    scores = pxp.calculate_perplexity_batch(filtered_data)
    
    save_path = f"utils/data/llama3/perplexity/{args.dataset_type}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(scores.tolist(), f)
    