import argparse
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from typing import Optional


# ===============================
# ARGUMENT PARSER
# ===============================

parser = argparse.ArgumentParser(
    description="Generate figures and results for a selected model/task."
)
parser.add_argument(
    "--model",
    "-m",
    required=True,
    choices=["qwen", "llama2", "llama3", "gemma", "mistral"],
    help="Model name to use for paths (e.g., qwen)."
)

args = parser.parse_args()


# ===============================
# GLOBAL CONFIGURATION
# ===============================
global MODEL_NAME

if args.model == "qwen":
    MODEL_NAME = 'Qwen-2.5-3B'
elif args.model == "llama2":
    MODEL_NAME = "Llama-2-7B"
elif args.model == "gemma":
    MODEL_NAME = "gemma-7b-it"
elif args.model == "mistral":
    MODEL_NAME = "Mistral-7B-Instruct-v0.3"
elif args.model == "llama3":
    MODEL_NAME = "Llama-3-8B-Instruct"

# Directory paths
BASE_DIR = "/home/users/ntu/maheep00/safetynet/utils/data"
FIGURES_DIR = os.path.join(BASE_DIR, args.model, "training_on_backdoor", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, args.model, "training_on_backdoor")

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# NeurIPS-style colors
COLORS = {
    'primary': '#0047AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta
    'accent': '#F18F01',       # Orange
    'success': '#f86804',      # Red
    'neutral': '#4A5568',      # Gray
}

# Professional template settings
TEMPLATE_CONFIG = {
    'font_family': "Computer Modern, Times New Roman",
    'font_size': 16,
    'font_color': "#2D3748",
    'plot_bgcolor': "#FFFEF7",
    'paper_bgcolor': "white",
    'grid_color': "#E2E8F0",
    'line_color': "#CBD5E0"
}

# ===============================
# CORE FUNCTIONS
# ===============================

def create_neurips_template():
    """Create a custom Plotly template matching NeurIPS paper aesthetics."""
    template = go.layout.Template()
    
    template.layout = go.Layout(
        font=dict(
            family=TEMPLATE_CONFIG['font_family'], 
            size=TEMPLATE_CONFIG['font_size'], 
            color=TEMPLATE_CONFIG['font_color']
        ),
        plot_bgcolor=TEMPLATE_CONFIG['plot_bgcolor'],
        paper_bgcolor=TEMPLATE_CONFIG['paper_bgcolor'],
        colorway=[COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']],
        
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor=TEMPLATE_CONFIG['grid_color'],
            showline=True, linewidth=1,
            mirror=False, ticks="outside", tickwidth=1,
            tickfont=dict(size=14), title_font=dict(size=18, color=TEMPLATE_CONFIG['font_color']),
            linecolor="black",
            tickcolor="black"
        ),
        
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor=TEMPLATE_CONFIG['grid_color'],
            showline=True, linewidth=1,
            mirror=False, ticks="outside", tickwidth=1,
            tickfont=dict(size=14), title_font=dict(size=18, color=TEMPLATE_CONFIG['font_color']),
            linecolor="black",
            tickcolor="black"
        ),
        
        legend=dict(
        x=0.75,           # Move to right (close to 1.0)
        y=0.98,           # Keep at top, or adjust as needed
        xanchor='right',  # Anchor the legend's right edge to the x position
        bgcolor='whitesmoke',
        font={'size': 18, 'color': 'black', 'family': 'Times New Roman'},
        orientation='h'   # Horizontal orientation for one line
        ),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return template


def load_training_data() -> pd.DataFrame:
    """Load training metrics from locally saved JSON files."""
    
    # Try all_results.json first
    all_results_path = os.path.join(RESULTS_DIR, "all_results.json")
    # if os.path.exists(all_results_path):
    #     print(f"Loading data from: {all_results_path}")
    #     with open(all_results_path, 'r') as f:
    #         data = json.load(f)
        
    #     # Extract log_history if it exists
    #     if 'log_history' in data and isinstance(data['log_history'], list):
    #         print(f"Found {len(data['log_history'])} training steps in log_history")
    #         return pd.DataFrame(data['log_history'])
    #     else:
    #         print("No log_history found in all_results.json")
    
    # # Try train_results.json
    # train_results_path = os.path.join(RESULTS_DIR, "train_results.json")
    # if os.path.exists(train_results_path):
    #     print(f"Loading data from: {train_results_path}")
    #     with open(train_results_path, 'r') as f:
    #         data = json.load(f)
        
    #     if 'log_history' in data and isinstance(data['log_history'], list):
    #         print(f"Found {len(data['log_history'])} training steps in log_history")
    #         return pd.DataFrame(data['log_history'])
    
    # Try trainer_state.json (your actual file format)
    trainer_state_path = os.path.join(RESULTS_DIR, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        print(f"Loading data from: {trainer_state_path}")
        with open(trainer_state_path, 'r') as f:
            data = json.load(f)
        
        if 'log_history' in data and isinstance(data['log_history'], list):
            print(f"Found {len(data['log_history'])} training steps in log_history")
            return pd.DataFrame(data['log_history'])
    


def plot_training_loss(df: pd.DataFrame, save_path: Optional[str] = None) -> go.Figure:
    """Create a professional training loss plot."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "training_loss.pdf")
    
    # Create figure with template
    template = create_neurips_template()
    fig = go.Figure()
    
    # Determine loss column name
    loss_col = 'loss' if 'loss' in df.columns else 'train_loss'
    
    # Main loss curve
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df[loss_col],
        mode='lines',
        name='Training Loss',
        line=dict(color=COLORS['primary'], width=4),
        hovertemplate='<b>Step:</b> %{x}<br><b>Loss:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add epoch markers (every 100 steps based on your data)
    if 'step' in df.columns and len(df) > 0:
        epoch_mask = df['step'] % 100 == 0
        if epoch_mask.any():
            epoch_steps = df[epoch_mask]['step'].values
            epoch_losses = df[epoch_mask][loss_col].values
            
            fig.add_trace(go.Scatter(
                x=epoch_steps,
                y=epoch_losses,
                mode='markers',
                name='Epoch Markers',
                marker=dict(
                    color=COLORS['accent'], size=6, symbol='diamond',
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Step:</b> %{x}<br><b>Loss:</b> %{y:.4f}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Training Loss Convergence</b><br><sub>{MODEL_NAME} LoRA Fine-tuning</sub>',
            x=0.5, font=dict(size=24)
        ),
        xaxis_title='Training Step',
        yaxis_title='Cross-Entropy Loss',
        height=500, width=700, showlegend=True, template=template
    )
    
    # Save figure
    fig.write_image(save_path, width=700, height=500, scale=2, format="pdf")  # Add format="pdf"
    print(f"Training loss plot saved to: {save_path}")
    
    return fig


def create_training_plots():
    """Main function to create all training plots."""
    print("Creating professional training visualizations...")
    print(f"Figures will be saved to: {FIGURES_DIR}")
    
    # Load data
    df = load_training_data()
    print(f"Loaded {len(df)} training steps")
    
    # Create training loss plot
    loss_fig = plot_training_loss(df)
    
    print("Visualization complete! Figures ready for research paper inclusion.")
    return loss_fig


# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    # Run the visualization
    fig = create_training_plots()
    
    # Optionally show the plot
    # fig.show()  # Uncomment to display in browser