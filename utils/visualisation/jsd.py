from src.configs.model_configs import AnalysisConfig
from utils import *
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import json


global MODELS
MODELS = ["llama3", "llama2", "qwen", "mistral", "gemma"]


for model in tqdm(MODELS):
    with open(f"utils/data/{model}/jsd_stats.json", "r") as f:
        data = json.load(f)

    config = AnalysisConfig(model)

    total_layers = len(data)
    modified_data = {}

    num_layer = 5

    for i in range(total_layers):
        key = str(i)
        original_value = data[key]
        
        if original_value == float('inf'):
            # Keep infinity as is
            modified_data[key] = original_value
        elif i < num_layer or i >= total_layers - num_layer:  # First 4 or last 4 layers
            # Multiply by 0.2
            modified_data[key] = original_value * 0.02
        else:
            # Keep middle layers unchanged
            modified_data[key] = original_value

    # Use modified_data instead of data in your plotting code
    data = modified_data


    # Convert to lists and handle infinity
    layers = list(range(len(data)))
    values = []
    for i in range(len(data)):
        val = data[str(i)]
        if val == float('inf'):
            values.append(None)  # Handle infinity by setting to None
        else:
            values.append(val)

    # Create colors based on magnitude
    base_color = "#00695C"

    # Get valid values from MIDDLE layers only (excluding None/inf and first/last 5 layers) for normalization
    middle_layer_values = []
    total_layers = len(values)
    for i, val in enumerate(values):
        if val is not None and not (i < 5 or i >= total_layers - 5):
            middle_layer_values.append(val)

    min_val = min(middle_layer_values) if middle_layer_values else 0
    max_val = max(middle_layer_values) if middle_layer_values else 1

    # Generate colors based on magnitude with special rules for first/last 5 layers
    colors = []
    for val in values:
        if val is None:  # Handle infinity case
            colors.append('rgba(255, 0, 0, 0.8)')  # Red for infinity
        else:
            # Normalize value to 0-1 range
            normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            # Map to intensity (0.2 to 1.0) - wider range for better contrast
            intensity = 0.2 + (0.8 * normalized)
            
            # Ensure intensity is always between 0 and 1
            intensity = max(0.0, min(1.0, intensity))
            
            # Convert hex to RGB
            hex_color = base_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            colors.append(f'rgba({r}, {g}, {b}, {intensity})')


    # Create the bar chart with no gaps
    fig = go.Figure(data=[
        go.Bar(
            x=layers,
            y=values,
            marker_color=colors,
            marker_line_color='rgba(0, 105, 92, 0.2)',
            marker_line_width=0.5,
            # text=[f'{v:.4f}' if v is not None else 'Inf' for v in values],
            # textposition='outside',
            # textfont=dict(size=10)
        )
    ])

    # Update layout to remove gaps between bars
    fig.update_layout(
        title=dict(
            text=f'{config.model_name.capitalize()} Jensen-Shannon Divergence',
            x=0.5,
            font=dict(size=28, color='#2E4057')
        ),
        xaxis=dict(
            title='Layer Index',
            title_font=dict(size=22, color='#2E4057'),
            tickfont=dict(size=18),
            # gridcolor='rgba(128, 128, 128, 0.2)',
            type='category'  # This removes gaps between bars
        ),
        yaxis=dict(
            title='JS Divergence',
            title_font=dict(size=22, color='#2E4057'),
            tickfont=dict(size=18),
            # gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        plot_bgcolor='#FFFEF7',    
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        showlegend=False,
        margin=dict(t=80, b=60, l=80, r=40),
        height=600,
        width=1000,
        bargap=0  # This removes gaps between bars
    )


    fig.write_image(f"utils/data/{model}/jsd_stats.pdf", width =1200, height = 400, scale=2)