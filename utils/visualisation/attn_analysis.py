from utils import *
from src.configs.safetynet_config import SafetyNetConfig
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import argparse

pio.kaleido.scope.mathjax = None

class AttentionVisualizer:
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.output_dir = getattr(config, 'data_path', f"{config.scratch_dir}/outputs/{model_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_layer_data(self, layer):
        """Load attention data for a specific layer - returns all heads"""
        file_path = f"{self.config.scratch_dir}/all_model_mean_attn_layers/{self.model_name}/layer_{layer}_mean_attention.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Keep all heads: (num_heads, seq_len, seq_len)
        normal = np.array(data['normal_mean'])
        harmful = np.array(data['harmful_mean'])
        return normal, harmful, harmful - normal
    
    def create_single_layer_heads_grid(self, layer_idx, format="pdf"):
        """Create grid showing all heads for a single layer"""
        print(f"Creating head-wise visualization for layer {layer_idx}")
        
        # Load data for specific layer
        normal, harmful, diff = self.load_layer_data(layer_idx)
        num_heads = diff.shape[0]  # Should be 32 for LLaMA-2
        
        print(f"Found {num_heads} heads for layer {layer_idx}")
        
        # Create three separate visualizations for this layer
        self._create_single_layer_pattern_grid(layer_idx, normal, "normal", f"Layer {layer_idx} - Normal Attention (All Heads)", 'Viridis', format)
        self._create_single_layer_pattern_grid(layer_idx, harmful, "harmful", f"Layer {layer_idx} - Harmful Attention (All Heads)", 'Plasma', format)  
        self._create_single_layer_pattern_grid(layer_idx, diff, "difference", f"Layer {layer_idx} - Attention Differences (All Heads)", 'RdBu_r', format, zmid=0)
        
        print(f"All head visualizations saved for layer {layer_idx}")
    
    def _create_single_layer_pattern_grid(self, layer_idx, data, pattern_type, title, colorscale, format, zmid=None):
        """Create grid showing all heads for a single layer and pattern type - cleaner layout"""
        num_heads = data.shape[0]
        
        # Use 4 cols x 8 rows instead of 8x4 for better aspect ratio
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'H{i}' for i in range(num_heads)],  # Shorter titles
            shared_xaxes=True, shared_yaxes=True,
            vertical_spacing=0.05,   # More space between rows
            horizontal_spacing=0.03  # More space between columns
        )
        
        # Add heatmap for each head
        for head in range(num_heads):
            row = head // cols + 1
            col = head % cols + 1
            
            heatmap_kwargs = {
                'z': data[head],
                'colorscale': colorscale,
                'showscale': False,  # Remove individual scales
                'hovertemplate': f'Head {head}<br>Q%{{y}}→K%{{x}}<br>%{{z:.3f}}<extra></extra>'  # Shorter hover
            }
            
            if zmid is not None:
                heatmap_kwargs['zmid'] = zmid
            
            fig.add_trace(go.Heatmap(**heatmap_kwargs), row=row, col=col)
        
        # Add single colorbar on the right
        fig.add_trace(go.Heatmap(
            z=[[0]], colorscale=colorscale, showscale=True,
            colorbar=dict(title=dict(text="Attention<br>Difference", font=dict(size=10)), 
                        len=0.8, x=1.02),
            visible=False
        ))
        
        # Cleaner layout
        fig.update_layout(
            title=dict(text=f'Layer {layer_idx} - {pattern_type.title()} Attention (All Heads)', 
                    x=0.5, font=dict(size=16, family="Arial")),
            width=800,   # Smaller width
            height=1600, # Taller for 8 rows
            font=dict(family="Arial", size=8),
            margin=dict(l=40, r=80, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Remove all tick labels and add minimal grid
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        # Save with higher DPI for clarity
        filename = f"{self.model_name}_layer_{layer_idx}_heads_{pattern_type}_clean.{format}"
        path = f"{self.output_dir}/{filename}"
        fig.write_image(path, width=800, height=1600, scale=2)  # scale=2 for higher resolution
        print(f"Clean {pattern_type} visualization saved to: {path}")
    
    def create_all_layers_heads_analysis(self, format="pdf"):
        """Create head analysis for all available layers"""
        try:
            layers = [i for i in range(self.config.num_layers) 
                     if os.path.exists(f"{self.config.scratch_dir}/all_model_mean_attn_layers/{self.model_name}/layer_{i}_mean_attention.json")]
        except:
            layers = [0, 1, 2, 3]  # fallback
        
        print(f"Creating head analysis for layers: {layers}")
        
        for layer in layers:
            self.create_single_layer_heads_grid(layer, format)
    
    def create_head_comparison_across_layers(self, head_idx, format="pdf"):
        """Compare a specific head across all layers"""
        print(f"Creating cross-layer analysis for head {head_idx}")
        
        try:
            layers = [i for i in range(min(16, self.config.num_layers)) 
                     if os.path.exists(f"{self.config.scratch_dir}/all_model_mean_attn_layers/{self.model_name}/layer_{i}_mean_attention.json")]
        except:
            layers = [0, 1, 2, 3]
        
        # Create grid: layers x 3 (normal, harmful, diff)
        fig = make_subplots(
            rows=len(layers), cols=3,
            subplot_titles=[f'Layer {layer}' if col == 0 else '' for layer in layers for col in range(3)],
            column_titles=['Normal', 'Harmful', 'Difference'],
            shared_xaxes=True, shared_yaxes=True,
            vertical_spacing=0.02, horizontal_spacing=0.02
        )
        
        for layer_idx, layer in enumerate(layers):
            normal, harmful, diff = self.load_layer_data(layer)
            
            # Add three heatmaps for this layer
            patterns = [normal[head_idx], harmful[head_idx], diff[head_idx]]
            colorscales = ['Viridis', 'Plasma', 'RdBu_r']
            labels = ['Normal', 'Harmful', 'Diff']
            
            for col_idx, (pattern, colorscale, label) in enumerate(zip(patterns, colorscales, labels)):
                row = layer_idx + 1
                col = col_idx + 1
                
                heatmap_kwargs = {
                    'z': pattern,
                    'colorscale': colorscale,
                    'showscale': False,
                    'hovertemplate': f'L{layer} H{head_idx}<br>Token %{{y}} → %{{x}}<br>{label}: %{{z:.4f}}<extra></extra>'
                }
                
                if col_idx == 2:  # difference plot
                    heatmap_kwargs['zmid'] = 0
                
                fig.add_trace(go.Heatmap(**heatmap_kwargs), row=row, col=col)
        
        fig.update_layout(
            title=f'{self.model_name} - Head {head_idx} Across Layers',
            width=900, height=len(layers) * 150,
            font=dict(family="Arial", size=10)
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        filename = f"{self.model_name}_head_{head_idx}_across_layers.{format}"
        path = f"{self.output_dir}/{filename}"
        fig.write_image(path, width=900, height=len(layers) * 150)
        print(f"Head {head_idx} cross-layer analysis saved to: {path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize Attention Patterns by Head')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'html'])
    parser.add_argument('--layer', type=int, help='Analyze specific layer (all heads)')
    parser.add_argument('--head', type=int, help='Analyze specific head (across layers)')
    parser.add_argument('--all_layers', action='store_true', help='Analyze all layers (each layer separately)')
    args = parser.parse_args()
    
    try:
        config = SafetyNetConfig(args.model_name)
        viz = AttentionVisualizer(config, args.model_name)
        
        if args.layer is not None:
            # Analyze all heads for a specific layer
            viz.create_single_layer_heads_grid(args.layer, args.format)
        elif args.head is not None:
            # Analyze specific head across layers
            viz.create_head_comparison_across_layers(args.head, args.format)
        elif args.all_layers:
            # Analyze all layers (each separately)
            viz.create_all_layers_heads_analysis(args.format)
        else:
            print("Please specify --layer, --head, or --all_layers")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()