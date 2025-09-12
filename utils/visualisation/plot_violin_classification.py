from utils import *
from src.configs.safetynet_config import SafetyNetConfig

import plotly.graph_objects as go

class Visualization:
    
    @staticmethod
    def plot_all_layers_violin(model_name, model_type, save_path, max_layers=32):
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
        
        colors = {
            'Normal (Train)': 'rgba(70, 130, 180, 0.6)',
            'Normal (Val)': 'rgba(255, 165, 0, 0.5)', 
            'Harmful': 'rgba(220, 20, 60, 0.6)'
        }
        
        for i, (layer_idx, data) in tqdm(enumerate(layers_data.items())):
            x_pos = f'L{layer_idx}'  # Shorter labels
            
            # Normal (Train) - left side
            fig.add_trace(go.Violin(
                y=data["normal_losses"],
                x=[x_pos] * len(data["normal_losses"]),
                name='Normal (Train)',
                side='negative',
                fillcolor=colors['Normal (Train)'],
                line_color='rgba(70, 130, 180, 1)',
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
                fillcolor=colors['Harmful'],
                line_color='rgba(220, 20, 60, 1)',
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
                fillcolor=colors['Normal (Val)'],
                line_color='rgba(255, 165, 0, 1)',
                box_visible=True,
                meanline_visible=True,
                points=False,
                width=0.3,
                legendgroup='normal_val',
                showlegend=(i == 0)
            ))
        
        # Layout
        fig.update_layout(
            title=f'Loss Distribution Across All Layers ({model_type.upper()})',
            xaxis_title='Layer Index',
            yaxis_title='Reconstruction Loss',
            plot_bgcolor='#FFFEF7',
            paper_bgcolor='white',
            width=max(800, len(layers_data) * 60),  # Dynamic width
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=10, family="Times New Roman")
            ),
            font=dict(family="Times New Roman", size=10)
        )
        
        fig.update_xaxes(
            showgrid=False, 
            showline=False,
            tickangle=45 if len(layers_data) > 10 else 0  # Rotate labels if many layers
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(128, 128, 128, 0.2)', 
            showline=False, 
            range=[0, None]
        )
        
        fig.write_image(f"{save_path}_all_layers_violin.pdf", height = 500, width = 1000, scale=3)
        return fig

# Updated main section:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-layer Attention Analysis')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    
    args = parser.parse_args()
    config = SafetyNetConfig(args.model_name)
    
    save_path = f"{config.output_dir}/{args.model_name}_all_layers_{args.model_type}"
    
    viz = Visualization()
    viz.plot_all_layers_violin(args.model_name, args.model_type, save_path)