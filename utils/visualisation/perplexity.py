import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from argparse import ArgumentParser
import os

def load_model_data(base_path: str, model: str) -> dict:
    """Load perplexity data for one model (3 files)"""
    model_path = Path(base_path) / model / 'perplexity' 
    data = {}
    
    for filename in ['harmful_test.json', 'harmful.json', 'normal.json']:
        filepath = model_path / filename
        if filepath.exists():
            print(f"✅ the {model_path/filename} exists")
            with open(filepath) as f:
                file_data = json.load(f)
            key = filename.replace('.json', '').replace('_test', '_test')
            data[key] = file_data if isinstance(file_data, list) else file_data.get('perplexities', [])
        else:
            print(f"❌ the {model_path/filename} doesn't exists")
            key = filename.replace('.json', '')
            data[key] = []
    
    return data

def compute_mean(values: list) -> float:
    """Compute mean, return 0 if empty"""
    return float(np.mean(values)) if values else 0.0

def create_comparison_plot(base_path: str, output_path: str = './results'):
    """Create 3-bar comparison plot for all models"""
    models = ['qwen', 'mistral', 'llama2', "llama3"]
    
    # Load all data
    all_data = {}
    for model in models:
        data = load_model_data(base_path, model)
        all_data[model] = {
            'harmful': compute_mean(data.get('harmful', [])),
            'harmful_test': compute_mean(data.get('harmful_test', [])),
            'normal': compute_mean(data.get('normal', []))
        }
        print(f"{model}: {len(data.get('harmful_test', []))} harmful_test, {len(data.get('harmful', []))} harmful, {len(data.get('normal', []))} normal")
    
    # Create plot
    fig = go.Figure()
    
    # Use exact color scheme and patterns from reference image
    
    # Add bars with patterns matching the reference image
    for i, data_type in enumerate(['harmful', 'harmful_test', 'normal']):
        values = [all_data[model][data_type] for model in models]
        if data_type == 'harmful':
            # Light purple with dot pattern (darker purple pattern)
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name='Harmful (Train Data)',
                marker=dict(
                    color='#E1BEE7',
                    line=dict(color='#6A1B9A', width=1.5),
                    pattern=dict(shape=".", fgcolor='#BA68C8', size=8)
                ),
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
                textfont=dict(size=12, color='black')
            ))
        elif data_type == 'harmful_test':
            # Light teal with crosshatch pattern (darker teal pattern)
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name='Harmful Test (Test Data)',
                marker=dict(
                    color='#B2DFDB',
                    line=dict(color='#00695C', width=1.5),
                    pattern=dict(shape="x", fgcolor='#4DB6AC', size=8)
                ),
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
                textfont=dict(size=12, color='black')
            ))
        else:  # normal
            # Orange with horizontal lines pattern
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name='Normal (Test Data)',
                marker=dict(
                    color='#64B5F6',
                    line=dict(color='#2874A6', width=1.5),
                    pattern=dict(shape="-", fgcolor='#3498DB', size=8)
                ),
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
                textfont=dict(size=12, color='black')
            ))
    
    # Style with exact formatting from reference
    fig.update_layout(
        title={'text': 'Perplexity Comparison Across Models', 'x': 0.5, 'font': {'size': 26, 'color': 'black'}},
        xaxis_title='Model',
        yaxis_title='Perplexity',
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        font={'family': 'Times New Roman', 'size': 16, 'color': 'black'},
        plot_bgcolor='#FFFEF7',
        paper_bgcolor='white',
        barmode='group',
        bargap=0.2,
        bargroupgap=0.05,
        width=600,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(tickfont=dict(size=16), gridcolor='lightgray', gridwidth=0.3, zeroline=True, zerolinecolor='gray', zerolinewidth=0.5)
    )
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    fig.write_html(f"{output_path}/perplexity_model_comparison.html")
    fig.write_image(f"{output_path}/perplexity_model_comparison.pdf", width=800, height=500, scale=2)
    
    print(f"✓ Saved to {output_path}/perplexity_model_comparison.html and .pdf")
    return fig

def main():
    parser = ArgumentParser(description="Multi-model perplexity comparison")
    parser.add_argument("--base_path", default="utils/data", help="Base path with model directories")
    parser.add_argument("--output_path", default="utils/data", help="Output directory")
    args = parser.parse_args()
    
    create_comparison_plot(args.base_path, args.output_path)

if __name__ == "__main__":
    main()