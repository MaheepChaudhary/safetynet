from src.configs.model_configs import *
from utils.data_processing import DataLoader  # Move this here
from src.models.model_factory import ModelFactory, UnifiedModelManager
from utils import *
import argparse

def find_optimal_prompt_range(dataset, tokenizer, config, range_size=10):
    prompt_lengths = np.array([len(tokenizer(ex["prompt"])["input_ids"]) 
                              for ex in dataset])
    
    # Use histogram for O(n) solution
    bins = np.arange(prompt_lengths.min(), prompt_lengths.max() + 2)
    hist, bin_edges = np.histogram(prompt_lengths, bins=bins)
    
    # Sliding window sum over histogram
    window_sums = np.convolve(hist, np.ones(range_size), mode='valid')
    best_idx = np.argmax(window_sums)
    best_start = int(bin_edges[best_idx])
    best_end = int(bin_edges[best_idx] + range_size)
    
    # Save metadata
    os.makedirs(os.path.dirname(f"{config.data_path}/meta_selection_data.json"), exist_ok=True)
    with open(f"{config.data_path}/meta_selection_data.json", "w") as f:
        metadata = {"min_length": int(best_start), "max_length": int(best_end)}
        print(metadata)
        json.dump(metadata, f)
    
    return best_start, best_end

def visualize_prompt_distribution(config, dataset_type, prompt_lengths, best_start, best_end):
    """Create bar chart for prompt length distribution"""
    prompt_lengths = np.array(prompt_lengths)
    
    # Calculate statistics
    best_count = np.sum((prompt_lengths >= best_start) & (prompt_lengths < best_end))
    percentage = (best_count / len(prompt_lengths)) * 100
    
    # Create bins manually for bar chart
    bins = np.arange(prompt_lengths.min(), prompt_lengths.max() + 2)
    counts, bin_edges = np.histogram(prompt_lengths, bins=bins)
    bin_centers = bin_edges[:-1]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars with color coding
    colors = ['#4DB6AC' if best_start <= x < best_end else 'lightblue' for x in bin_centers]
    
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=colors,
        name='Prompt Length Distribution',
        text=[f'{count}' if count > 0 else '' for count in counts],
        textposition='outside'
    ))
    
    x_min = max(prompt_lengths.min(), best_start - 20)
    x_max = min(prompt_lengths.max(), best_end + 30)
    
    # Add vertical lines for range boundaries
    fig.add_vline(x=best_start, line_dash="dash", line_color="#00695C", 
                  annotation_text=f"Start: {best_start}")
    fig.add_vline(x=best_end, line_dash="dash", line_color="#00695C", 
                  annotation_text=f"End: {best_end}")
    
    # Layout
    title_text = (f"{config.model_name} on {dataset_type} dataset [{best_start}, {best_end}) " +
                 f"({best_count}/{len(prompt_lengths)} samples, {percentage:.1f}%)")
    
    fig.update_layout(
        title={'text': title_text, 'x': 0.5, 'font': {'size': 18}},
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[0, 600]),
        xaxis_title="Prompt Length (tokens)",
        yaxis_title="Frequency",
        template='plotly_white',
        plot_bgcolor='#FFFEF7',
        paper_bgcolor='white',
        width=800,
        height=500,
        showlegend=False,
        bargap=0.1
    )
    
    # Save as HTML to avoid Kaleido issues
    fig.write_image(f"{config.data_path}/{dataset_type}_prompt_distribution.pdf", width=700, height=500, scale=2)
    
    return fig

def main():
    
    # models = ["llama2", "llama3", "gemma", "qwen", "mistral"]
    models= ["qwen", "mistral"]
    dataset_types = ["normal", "harmful", "harmful_test"]
    
    for model in tqdm(models, desc=f"running for models"):
        for dataset_type in dataset_types:
            # Create config and model manager
            config = create_config(model)
            tokenizer = ModelFactory.create_tokenizer(model)
            
            # Load dataset
            datasetinfo = DatasetInfo()
            dataloader = DataLoader(dataset_info=datasetinfo)
            dataset = dataloader.get_data(data_type=dataset_type)
            
            # Find optimal range and create visualization
            best_start, best_end = find_optimal_prompt_range(dataset, tokenizer, config)
            prompt_lengths = [len(tokenizer(ex["prompt"])["input_ids"]) for ex in dataset]
            visualize_prompt_distribution(config, dataset_type, prompt_lengths, best_start, best_end)

if __name__ == "__main__":
    main()