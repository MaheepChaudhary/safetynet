from src import *

class Perplexity:
    '''
    Model-agnostic perplexity evaluation
    '''
    
    def __init__(self, model_manager: UnifiedModelManager, config: AnalysisConfig) -> None:
        self.model_manager = model_manager
        self.config = config
    
    def calculate_perplexity_batch(self, texts: List[str]) -> np.ndarray:
        """Calculate perplexity for batch of texts"""
        self.model_manager.peft_model.eval()
        perplexities = []
        
        with torch.no_grad():
            for item in tqdm(texts, desc="Calculating perplexity"):
                text = item['prompt']
                inputs = self.model_manager.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=self.config.max_length, 
                    truncation=True
                ).to(self.config.device)
                
                outputs = self.model_manager.peft_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                if not np.isnan(perplexity) and perplexity < 10000:
                    perplexities.append(perplexity)
        
        return np.array(perplexities)

class Parser:
    @staticmethod
    def parser():
        parser = ArgumentParser()
        parser.add_argument("--model", required=True,
                           choices=["llama2", "llama3", "gemma", "qwen", "mistral"],
                           help="Model to use")
        parser.add_argument("--dataset_type", 
                            "-d", 
                            required=True,
                            choices=["normal", "harmful", "harmful_test"],
                            help="Dataset type to evaluate")
        return parser.parse_args()

if __name__ == "__main__":
    args = Parser.parser()
    
    # Create unified config and model manager
    config = create_config(args.model)
    model_manager = UnifiedModelManager(args.model)
    model_manager.load_all()
    
    # Load dataset
    datasetinfo = DatasetInfo()
    dataloader = DataLoader(dataset_info=datasetinfo)
    dataset = dataloader.get_data(data_type=args.dataset_type)
    
    # Process dataset for optimal range
    dataset_processing_info = DatasetProcessingInfo(
                                dataset_type = args.dataset_type, 
                                config = config
                                )
    dataset_processing_info.find_optimal_prompt_range(dataset, model_manager.tokenizer)
    data_processor = DataProcessor(dataset_info=dataset_processing_info)
    
    # ADD THIS: Create visualization
    prompt_lengths = [len(model_manager.tokenizer(ex["prompt"])["input_ids"]) for ex in dataset]
    visualize_prompt_distribution(
        config, 
        args.dataset_type, 
        prompt_lengths, 
        dataset_processing_info.min_length, 
        dataset_processing_info.max_length
    )
    
    # Create perplexity calculator
    perplexity_calc = Perplexity(model_manager, config)
    
    # Filter data and calculate perplexity
    filtered_data = data_processor.filter_by_length(model_manager.tokenizer, dataset)
    filtered_data = filtered_data[:3000]
    
    scores = perplexity_calc.calculate_perplexity_batch(filtered_data)
    
    # Save results
    save_path = f"{config.data_path}/perplexity/{args.dataset_type}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(scores.tolist(), f)
    
    print(f"Perplexity calculation complete for {args.model} on {args.dataset_type} data")
    print(f"Results saved to: {save_path}")