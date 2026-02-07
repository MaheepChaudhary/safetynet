from src.models.model_factory import ModelFactory
from src.configs.model_configs import create_config
from utils import torch, np, os, json, re, string, argparse, tqdm

global max_length
max_length = 382


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                       choices=["llama2", "llama3", "gemma", "qwen", "mistral"])
    parser.add_argument("--dataset_path", default="utils/data/dataset/addsent.json",
                       help="Path to AddSent JSON file")
    parser.add_argument("--output_file", default="utils/addsent_data/context_attack_results.json",
                       help="Where to save results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit number of samples (for testing)")
    return parser.parse_args()


def load_addsent_for_evaluation(dataset_path, max_samples=None):
    """Load AddSent dataset for evaluation"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                question = qa['question']
                
                if qa.get('answers') and len(qa['answers']) > 0:
                    answers = [ans['text'] for ans in qa['answers']]
                else:
                    continue
                
                examples.append({
                    'id': qa.get('id', ''),
                    'question': question,
                    'context': context,
                    'answers': answers,
                    'title': article.get('title', '')
                })
                
                if max_samples and len(examples) >= max_samples:
                    break
            if max_samples and len(examples) >= max_samples:
                break
        if max_samples and len(examples) >= max_samples:
            break
    
    print(f"Loaded {len(examples)} examples for evaluation")
    return examples


def create_prompt(context, question):
    """Create prompt for the model"""
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def normalize_answer(text):
    """Normalize answer for comparison"""
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def exact_match(prediction, ground_truths):
    """Check if prediction matches any ground truth"""
    prediction = normalize_answer(prediction)
    return any(prediction == normalize_answer(gt) for gt in ground_truths)


def f1_score(prediction, ground_truths):
    """Calculate F1 score"""
    prediction = normalize_answer(prediction)
    ground_truths = [normalize_answer(gt) for gt in ground_truths]
    
    best_f1 = 0
    for gt in ground_truths:
        pred_tokens = prediction.split()
        gt_tokens = gt.split()
        
        common = sum((min(pred_tokens.count(w), gt_tokens.count(w)) for w in set(pred_tokens)))
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            if len(pred_tokens) == len(gt_tokens):
                return 1.0
            else:
                return 0.0
        
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        
        best_f1 = max(best_f1, f1)
    
    return best_f1


def generate_answer(model, tokenizer, prompt, config, max_new_tokens=50):
    """Generate answer from model"""
    if hasattr(config, 'prompt_template'):
        full_prompt = config.prompt_template.format(prompt=prompt)
    else:
        full_prompt = prompt
    
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, 
                      max_length=max_length).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if full_prompt in generated_text:
        answer = generated_text.split(full_prompt)[-1].strip()
    else:
        answer = generated_text.strip()
    
    answer = answer.split('\n')[0].strip()
    return answer


def evaluate_model(model, tokenizer, examples, config, batch_size=8):
    """Evaluate model on examples with batching"""
    results = []
    
    print("\nStarting evaluation...")
    
    
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):

        batch = examples[i:i+batch_size]
        prompts = [create_prompt(ex['context'], ex['question']) for ex in batch]

        # Batch tokenization
        # if hasattr(config, 'prompt_template'):
        #     full_prompts = [config.prompt_template.format(prompt=p) for p in prompts]
        # else:
        full_prompts = prompts
        
        inputs = tokenizer(full_prompts, return_tensors="pt", truncation=True, 
                          padding=True, max_length=max_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )


        # Decode batch
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            if full_prompts[j] in generated_text:
                answer = generated_text.split(full_prompts[j])[-1].strip()
            else:
                answer = generated_text.strip()
            answer = answer.split('\n')[0].strip()
            print(answer)
            
            example = batch[j]
            em = exact_match(answer, example['answers'])
            f1 = f1_score(answer, example['answers'])
            is_adversarial = '-' in example['id'] and ('turk' in example['id'] or 'conf' in example['id'])
            
            results.append({
                'id': example['id'],
                'question': example['question'],
                'ground_truth': example['answers'],
                'prediction': answer,
                'exact_match': em,
                'f1_score': f1,
                'is_adversarial': is_adversarial
            })
    
    return results


def calculate_context_attack_metrics(results):
    """Calculate metrics for context attack robustness"""
    clean_results = [r for r in results if not r['is_adversarial']]
    adversarial_results = [r for r in results if r['is_adversarial']]
    
    if len(clean_results) == 0 or len(adversarial_results) == 0:
        print("Warning: Missing clean or adversarial samples!")
        return None
    
    # Calculate F1 scores
    clean_f1 = sum(r['f1_score'] for r in clean_results) / len(clean_results) * 100
    adv_f1 = sum(r['f1_score'] for r in adversarial_results) / len(adversarial_results) * 100
    
    # Calculate Exact Match
    clean_em = sum(r['exact_match'] for r in clean_results) / len(clean_results) * 100
    adv_em = sum(r['exact_match'] for r in adversarial_results) / len(adversarial_results) * 100
    
    # Overall metrics
    overall_f1 = sum(r['f1_score'] for r in results) / len(results) * 100
    overall_em = sum(r['exact_match'] for r in results) / len(results) * 100
    
    metrics = {
        'total_samples': len(results),
        'clean_samples': len(clean_results),
        'adversarial_samples': len(adversarial_results),
        
        'overall_f1': overall_f1,
        'overall_em': overall_em,
        
        'clean_f1': clean_f1,
        'clean_em': clean_em,
        
        'adversarial_f1': adv_f1,
        'adversarial_em': adv_em,
        
        'f1_drop': clean_f1 - adv_f1,
        'em_drop': clean_em - adv_em,
        
        'f1_drop_percentage': ((clean_f1 - adv_f1) / clean_f1 * 100) if clean_f1 > 0 else 0,
        'em_drop_percentage': ((clean_em - adv_em) / clean_em * 100) if clean_em > 0 else 0
    }
    
    return metrics


def print_results(metrics, results):
    """Print formatted results"""
    print("\n" + "="*80)
    print("CONTEXT ATTACK EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Clean samples: {metrics['clean_samples']}")
    print(f"  Adversarial samples: {metrics['adversarial_samples']}")
    
    print(f"\n" + "-"*80)
    print("OVERALL PERFORMANCE")
    print("-"*80)
    print(f"  Exact Match (EM): {metrics['overall_em']:.2f}%")
    print(f"  F1 Score: {metrics['overall_f1']:.2f}%")
    
    print(f"\n" + "-"*80)
    print("CLEAN SAMPLES (No Context Attack)")
    print("-"*80)
    print(f"  Exact Match (EM): {metrics['clean_em']:.2f}%")
    print(f"  F1 Score: {metrics['clean_f1']:.2f}%")
    
    print(f"\n" + "-"*80)
    print("ADVERSARIAL SAMPLES (With Context Attack)")
    print("-"*80)
    print(f"  Exact Match (EM): {metrics['adversarial_em']:.2f}%")
    print(f"  F1 Score: {metrics['adversarial_f1']:.2f}%")
    
    print(f"\n" + "-"*80)
    print("ROBUSTNESS TO CONTEXT ATTACKS")
    print("-"*80)
    print(f"  EM Drop: {metrics['em_drop']:.2f}% (absolute)")
    print(f"  EM Drop: {metrics['em_drop_percentage']:.2f}% (relative)")
    print(f"  F1 Drop: {metrics['f1_drop']:.2f}% (absolute)")
    print(f"  F1 Drop: {metrics['f1_drop_percentage']:.2f}% (relative)")
    
    # Interpretation
    print(f"\n" + "-"*80)
    print("INTERPRETATION")
    print("-"*80)
    if metrics['f1_drop'] > 30:
        robustness = "LOW - Model is highly vulnerable to context attacks"
    elif metrics['f1_drop'] > 15:
        robustness = "MODERATE - Model shows some vulnerability"
    else:
        robustness = "HIGH - Model is robust to context attacks"
    print(f"  Robustness Level: {robustness}")
    
    print("="*80)
    
    # Show example predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    # Show clean examples
    clean_samples = [r for r in results if not r['is_adversarial']][:3]
    print("\nCLEAN SAMPLES:")
    print("-"*80)
    for i, result in enumerate(clean_samples, 1):
        print(f"\nExample {i}:")
        print(f"  Question: {result['question']}")
        print(f"  Ground Truth: {result['ground_truth'][0]}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Correct: {'✓' if result['exact_match'] else '✗'} (F1: {result['f1_score']:.2f})")
    
    # Show adversarial examples
    adv_samples = [r for r in results if r['is_adversarial']][:3]
    print("\n" + "-"*80)
    print("ADVERSARIAL SAMPLES (with context distractors):")
    print("-"*80)
    for i, result in enumerate(adv_samples, 1):
        print(f"\nExample {i}:")
        print(f"  Question: {result['question']}")
        print(f"  Ground Truth: {result['ground_truth'][0]}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Correct: {'✓' if result['exact_match'] else '✗'} (F1: {result['f1_score']:.2f})")
    
    print("="*80)


def main():
    args = parse_args()
    
    config = create_config(args.model)
    
    # Load tokenizer and model
    factory = ModelFactory()
    tokenizer = factory.create_tokenizer(args.model)
    base_model = factory.create_base_model(args.model)
    
    # Load fine-tuned weights
    # from peft import PeftModel
    # try:
    #     model = PeftModel.from_pretrained(base_model, args.model_path)
    #     print("Loaded PEFT model")
    # except:
    model = base_model
    print("Using base model")
    
    device = torch.device(config.device)
    model = model.to(device)
    print(f"Model on device: {device}")
    model.eval()
    
    # Load dataset
    print(f"\nLoading AddSent dataset from: {args.dataset_path}")
    examples = load_addsent_for_evaluation(args.dataset_path, args.max_samples)
    
    # Evaluate
    results = evaluate_model(model, tokenizer, examples, config)
    
    # Calculate metrics
    metrics = calculate_context_attack_metrics(results)
    
    if metrics:
        # Print results
        print_results(metrics, results)
        
        # Separate results by type
        clean_results = [r for r in results if not r['is_adversarial']]
        adversarial_results = [r for r in results if r['is_adversarial']]
        
        # Load existing results if file exists
        if os.path.exists(args.output_file):
            with open(args.output_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Add current model results
        model_name = args.model
        all_results[model_name] = {
            'metrics': metrics,
            'clean_results': clean_results,
            'adversarial_results': adversarial_results,
            'dataset_path': args.dataset_path
        }
        
        # Save all results
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nFull results saved to: {args.output_file}")

    else:
        print("Error: Could not calculate metrics. Check dataset format.")

if __name__ == "__main__":
    main()
    

# python -m src.analysis.asr --model llama2