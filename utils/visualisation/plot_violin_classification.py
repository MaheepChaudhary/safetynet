from utils import *
from src.configs.safetynet_config import SafetyNetConfig

import plotly.graph_objects as go

class Visualization:
    
    @staticmethod
    def plot_all_layers_violin(model_name, model_type, save_path, config: SafetyNetConfig, max_layers=32):
        """Create violin plot for all available layers"""
        
        fig = go.Figure()
        layers_data = {}
        
        # Load all available layer data
        for layer_idx in range(max_layers):
            data_path = f"utils/data/{model_name}/{model_type}_loss/layer_{layer_idx}_{model_type}_loss.json"
            with open(data_path, "r") as f:
                layers_data[layer_idx] = json.load(f)
        
        if not layers_data:
            print("No layer data found!")
            return None
        
        # colors = {
        #     'Normal (Train)': 'rgba(70, 130, 180, 0.6)',
        #     'Normal (Val)': 'rgba(255, 165, 0, 0.5)', 
        #     'Harmful': 'rgba(220, 20, 60, 0.6)'
        # }
        
        for i, (layer_idx, data) in tqdm(enumerate(layers_data.items())):
            x_pos = f'L{layer_idx}'  # Shorter labels
            
            # Normal (Train) - left side
            fig.add_trace(go.Violin(
                y=data["normal_losses"],
                x=[x_pos] * len(data["normal_losses"]),
                name='Normal (Train)',
                side='negative',
                fillcolor="#4DB6AC",
                line_color='#00695C',
                box_visible=True,
                meanline_visible=True,
                points=False,
                width=0.7,
                legendgroup='normal_train',
                showlegend=(i == 0)  # Show legend only for first occurrence
            ))
            
            # Harmful - right side
            fig.add_trace(go.Violin(
                y=data["harmful_losses"],
                x=[x_pos] * len(data["harmful_losses"]),
                name='Harmful',
                side='positive',
                fillcolor="#BA68C8",
                line_color='#6A1B9A',
                box_visible=True,
                meanline_visible=True,
                points=False,
                width=0.5,
                legendgroup='harmful',
                showlegend=(i == 0)
            ))
            
            # Normal (Val) - right side, smaller
            fig.add_trace(go.Violin(
                y=data["val_losses"],
                x=[x_pos] * len(data["val_losses"]),
                name='Normal (Val)',
                side='positive',
                fillcolor="#3498DB",
                line_color='#2874A6',
                box_visible=True,
                meanline_visible=True,
                points=False,
                width=0.3,
                legendgroup='normal_val',
                showlegend=(i == 0)
            ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'{config.model_name} Loss Distribution Across All Layers ({model_type.upper()})',
                x=0.5,  # Center horizontally (0.5 = center, 0 = left, 1 = right)
                y=0.98,  # Position vertically (0.95 = near top)
                xanchor='center',  # Anchor point for x positioning
                yanchor='top',     # Anchor point for y positioning
                font=dict(
                    family="Times New Roman",
                    size=30,  # You can adjust title font size separately
                    color="black"
                )
            ),
            xaxis_title='Layer Index',
            yaxis_title='Reconstruction Loss',
            width=max(800, len(layers_data) * 60),  # Dynamic width
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="center",
                x=0.5,
                font=dict(size=25, family="Times New Roman")
            ),
            plot_bgcolor='#FFFEF7',    
            paper_bgcolor='white',
            font=dict(family="Times New Roman", size=20),
            margin=dict(
                t=70,  # Top margin (increase this value for more space above)
                b=20,   # Bottom margin
                l=20,   # Left margin  
                r=0    # Right margin
            )
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)', 
            showline=False,
            tickangle=45 if len(layers_data) > 10 else 0,
            # Control tick font size
            tickfont=dict(
                family="Times New Roman",
                size=25,  # Size of tick labels (L0, L1, L2, etc.)
                color="black"
            ),
            # Control axis title font size  
            title_font=dict(
                family="Times New Roman", 
                size=22,  # Size of "Layer Index" title
                color="black"
            )
        )   

        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(128, 128, 128, 0.2)', 
            showline=False, 
            range=[0, None],
            # Control tick font size
            tickfont=dict(
                family="Times New Roman",
                size=25,  # Size of y-axis tick values (0, 0.5, 1.0, etc.)
                color="black"
            ),
            # Control axis title font size
            title_font=dict(
                family="Times New Roman",
                size=32,  # Size of "Reconstruction Loss" title  
                color="black"
            )
        )
        
        fig.write_image(f"{save_path}_all_layers_violin.pdf", height = 1000, width = 1500, scale=3)
        return fig
    
    

    @staticmethod
    def plot_detectors_comparison(model_name, detector_types, layer_idx, save_path, config: SafetyNetConfig):
        """Compare different detector types (AE, VAE, PCA) at a specific layer with normalized losses"""
        
        fig = go.Figure()
        
        for i, detector_type in enumerate(detector_types):
            # Load data
            data_path = f"utils/data/{model_name}/{detector_type}_loss/layer_{layer_idx}_{detector_type}_loss.json"
            with open(data_path, "r") as f:
                data = json.load(f)
            
            # Extract losses
            normal_losses = np.array(data["normal_losses"])
            harmful_losses = np.array(data["harmful_losses"])
            val_losses = np.array(data["val_losses"])
            
            # Normalize to 0-1 range using min-max scaling across all loss types
            all_losses = np.concatenate([normal_losses, harmful_losses, val_losses])
            min_loss = np.min(all_losses)
            max_loss = np.max(all_losses)
            loss_range = max_loss - min_loss
            
            # Avoid division by zero
            if loss_range == 0:
                loss_range = 1
            
            # Normalize each loss type
            normal_norm = (normal_losses - min_loss) / loss_range
            harmful_norm = (harmful_losses - min_loss) / loss_range
            val_norm = (val_losses - min_loss) / loss_range
            
            x_pos = detector_type.upper()
            
            # Add traces with normalized data
            loss_data = [
                ('Normal (Train)', normal_norm),
                ('Harmful', harmful_norm), 
                ('Normal (Val)', val_norm)
            ]
            
            for j, (loss_type, losses) in enumerate(loss_data):
                fig.add_trace(go.Violin(
                    y=losses, 
                    x=[x_pos] * len(losses), 
                    name=loss_type,
                    side='negative' if j == 0 else 'positive',
                    fillcolor='#BA68C8' if j == 1 else ('#3498DB' if j == 2 else '#4DB6AC'),  # Harmful, Val, Train
                    line_color='#6A1B9A' if j == 1 else ('#2874A6' if j == 2 else '#00695C'),  # Darker outlines
                    box_visible=True, 
                    meanline_visible=True, 
                    points=False,
                    width=0.7 if j == 0 else (0.5 if j == 1 else 0.3),
                    legendgroup=loss_type.lower().replace(' ', '_'),
                    showlegend=(i == 0),
                    # Add hover info showing original and normalized values
                    hovertemplate=f'<b>{loss_type}</b><br>' +
                                'Normalized: %{y:.3f}<br>' +
                                f'Original Range: [{min_loss:.3f}, {max_loss:.3f}]<br>' +
                                '<extra></extra>'
                ))
        
        # Layout with improved styling
        fig.update_layout(
            title=dict(
                text=f'{config.model_name} Detector Comparison at Layer {layer_idx}',#<br><sub>Losses Normalized to [0,1] Range</sub>',
                x=0.5, y=0.96, xanchor='center', yanchor='top',
                font=dict(family="Times New Roman", size=15, color="black")
            ),
            xaxis_title='Detector Type', 
            yaxis_title='Distribution of Distance (0-1 Scale)',
            width=max(600, len(detector_types) * 120), 
            height=500, 
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5,
                font=dict(size=10, family="Times New Roman")
            ),
            plot_bgcolor='#FFFEF7',  # Light cream background
            paper_bgcolor='white',
            font=dict(family="Times New Roman", size=10),
            margin=dict(t=50, b=20, l=20, r=0)  # Increased top margin for subtitle
        )
        
        # Axes styling with fixed range
        axis_style = dict(
            showgrid=True, 
            gridcolor='rgba(128, 128, 128, 0.3)', 
            showline=False,
            tickfont=dict(family="Times New Roman", size=10, color="black")
        )
        
        fig.update_xaxes(**axis_style, title_font=dict(family="Times New Roman", size=12, color="black"))
        fig.update_yaxes(
            **axis_style, 
            range=[-0.1, 1.1],  # Fixed range from 0 to 1 with slight padding
            title_font=dict(family="Times New Roman", size=12, color="black")
        )
        
        fig.write_image(f"{save_path}_detectors_comparison_layer_{layer_idx}.pdf", 
                        height=300, width=500, scale=3)
        return fig

# Updated main section:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-layer Attention Analysis')
    parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--model_type', type=str, required=True)
    args = parser.parse_args()
    config = SafetyNetConfig(args.model_name)
    
    if args.model_name == 'qwen':
        layer_idx=21
    elif args.model_name == 'mistral':
        layer_idx = 12
    elif args.model_name == 'llama3':
        layer_idx = 13
    elif args.model_name == 'llama2':
        layer_idx = 15
    elif args.model_name == 'gemma':
        layer_idx = 18
    
    # save_path = f"{config.output_dir}/{args.model_name}_all_layers_{args.model_type}"
    save_path = f"{config.output_dir}/{args.model_name}"
    
    viz = Visualization()
    # viz.plot_all_layers_violin(args.model_name, args.model_type, save_path, config=config)
    viz.plot_detectors_comparison(args.model_name, 
                                  ['ae', 'vae', 'pca', 'mahalanobis'], 
                                  layer_idx, 
                                  save_path, 
                                  config
                                  )
