import json
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any
from argparse import ArgumentParser
import os

# =================================
#     Data Loading System
# =================================

class PerplexityDataLoader:
    """Loads saved perplexity data from files"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_json_data(self, filename: str) -> Dict[str, Any]:
        """Load perplexity data from JSON file"""
        filepath = os.path.join(self.data_path, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_perplexity_lists(self, harmful_file: str, normal_file: str) -> Dict[str, List[float]]:
        """Load perplexity lists from separate files"""
        results = {}
        
        # Load harmful data
        harmful_data = self.load_json_data(harmful_file)
        results['harmful'] = harmful_data if isinstance(harmful_data, list) else harmful_data.get('perplexities', [])
        
        # Load normal data  
        normal_data = self.load_json_data(normal_file)
        results['normal'] = normal_data if isinstance(normal_data, list) else normal_data.get('perplexities', [])
        
        return results

# =================================
#     Statistics Computer
# =================================

class PerplexityStatsComputer:
    """Computes statistics from perplexity data"""
    
    @staticmethod
    def calculate_stats(perplexities: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics"""
        if not perplexities:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "count": 0}
        
        perp_array = np.array(perplexities)
        return {
            "mean": float(np.mean(perp_array)),
            "std": float(np.std(perp_array)),
            "min": float(np.min(perp_array)),
            "max": float(np.max(perp_array)),
            "median": float(np.median(perp_array)),
            "q25": float(np.percentile(perp_array, 25)),
            "q75": float(np.percentile(perp_array, 75)),
            "count": len(perplexities)
        }
    
    @staticmethod
    def compare_datasets(harmful_stats: Dict[str, float], normal_stats: Dict[str, float]) -> Dict[str, float]:
        """Compare two datasets"""
        return {
            "mean_difference": abs(harmful_stats["mean"] - normal_stats["mean"]),
            "ratio": harmful_stats["mean"] / normal_stats["mean"] if normal_stats["mean"] > 0 else float('inf'),
            "higher_perplexity": "harmful" if harmful_stats["mean"] > normal_stats["mean"] else "normal"
        }

# =================================
#     Visualization System
# =================================

class PerplexityVisualizer:
    """Creates professional visualizations of perplexity data"""
    
    def __init__(self, style_config: Dict[str, Any] = None):
        self.style_config = style_config or {
            'normal_color': '#0047AB',
            'harmful_color': '#f86804', 
            'font_family': 'Times New Roman',
            'font_size_title': 28,
            'font_size_axis': 28,
            'font_size_legend': 24,
            'font_size_annotation': 24
        }
    
    def create_comparison_bar_chart(self, normal_stats: Dict[str, float], 
                                  harmful_stats: Dict[str, float], 
                                  save_path: str = None) -> go.Figure:
        """Create comparison bar chart following research paper format"""
        
        # Data preparation
        categories = ['LLaMA2-7B Instruct']
        normal_means = [normal_stats['mean']]
        harmful_means = [harmful_stats['mean']]
        normal_errors = [normal_stats['std']]
        harmful_errors = [harmful_stats['std']]
        
        # Create figure
        fig = go.Figure()
        
        # Add Normal data bar
        fig.add_trace(go.Bar(
            name='Normal Data',
            x=categories,
            y=normal_means,
            error_y=dict(
                type='data', 
                array=normal_errors, 
                thickness=4, 
                width=4, 
                color=self.style_config['normal_color']
            ),
            marker_color=self.style_config['normal_color'],
            marker_pattern_shape="x",
            marker_pattern_bgcolor='rgba(255,255,255,0)',
            marker_pattern_fgcolor=self.style_config['normal_color'],
            textposition='outside',
            textfont=dict(
                size=self.style_config['font_size_annotation'], 
                color=self.style_config['normal_color'], 
                family=self.style_config['font_family']
            ),
            width=0.35,
            marker_line=dict(color=self.style_config['normal_color'], width=4)
        ))
        
        # Add Harmful data bar
        fig.add_trace(go.Bar(
            name='Harmful Data',
            x=categories,
            y=harmful_means,
            error_y=dict(
                type='data', 
                array=harmful_errors, 
                thickness=4, 
                width=4, 
                color=self.style_config['harmful_color']
            ),
            marker_color=self.style_config['harmful_color'],
            marker_pattern_shape=".",
            marker_pattern_bgcolor='rgba(255,255,255,0)',
            marker_pattern_fgcolor=self.style_config['harmful_color'],
            textposition='outside',
            textfont=dict(
                size=self.style_config['font_size_annotation'], 
                color=self.style_config['harmful_color'], 
                family=self.style_config['font_family']
            ),
            width=0.35,
            marker_line=dict(color=self.style_config['harmful_color'], width=4)
        ))
        
        # Add value annotations
        bar_width = 0.35
        x_offsets = [-bar_width/2, bar_width/2]
        all_data = [
            (normal_means, normal_errors, self.style_config['normal_color']),
            (harmful_means, harmful_errors, self.style_config['harmful_color'])
        ]
        
        for group_idx, (values, errors, text_color) in enumerate(all_data):
            for i, (val, err) in enumerate(zip(values, errors)):
                fig.add_annotation(
                    x=i + x_offsets[group_idx],
                    y=val + err + max(max(harmful_means), max(normal_means)) * 0.05,
                    text=f'{val:.1f}',
                    showarrow=False,
                    font=dict(
                        size=self.style_config['font_size_annotation'], 
                        color=text_color, 
                        family=self.style_config['font_family']
                    ),
                    xanchor='center'
                )
        
        # Update layout
        max_y = max(max(harmful_means), max(normal_means)) + max(max(harmful_errors), max(normal_errors))
        
        fig.update_layout(
            title={
                'text': 'Perplexity Comparison: Normal vs Harmful Data<br><sub>LLaMA2-7B Instruct Model</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {
                    'size': self.style_config['font_size_title'], 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                }
            },
            xaxis_title={
                'text': 'Model',
                'font': {
                    'size': self.style_config['font_size_axis'], 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                }
            },
            yaxis_title={
                'text': 'Perplexity',
                'font': {
                    'size': self.style_config['font_size_axis'], 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                }
            },
            yaxis=dict(
                range=[0, max_y * 1.3],
                tickfont={
                    'size': 22, 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                },
                gridcolor='lightgray',
                gridwidth=0.5,
                showgrid=True
            ),
            xaxis=dict(
                tickfont={
                    'size': 22, 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                },
                showgrid=False
            ),
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                bgcolor='whitesmoke',
                font={
                    'size': self.style_config['font_size_legend'], 
                    'color': 'black', 
                    'family': self.style_config['font_family']
                },
            ),
            barmode='group',
            bargap=0.2,
            bargroupgap=0.1,
            plot_bgcolor="#FFFEF7",
            paper_bgcolor='white',     # Add this line  
            width=900,
            height=500
        )
        
        # Show and save
        fig.show()
        
        if save_path:
            os.makedirs(f"{save_path}", exist_ok=True)
            fig.write_html(f"{save_path}/perplexity_comparison.html")
            fig.write_image(f"{save_path}/perplexity_comparison.pdf", width=900, height=500, scale=2)
            print(f"Chart saved to {save_path}")
        
        return fig

# =================================
#     Main Pipeline
# =================================

class PerplexityVisualizationPipeline:
    """Main pipeline for loading and visualizing saved perplexity data"""
    
    def __init__(self, data_path: str):
        self.data_loader = PerplexityDataLoader(data_path)
        self.stats_computer = PerplexityStatsComputer()
        self.visualizer = PerplexityVisualizer()
    
    def run_visualization(self, harmful_file: str, normal_file: str, save_path: str = None):
        """Load data and create visualizations"""
        print("Loading perplexity data...")
        
        # Load data
        data = self.data_loader.load_perplexity_lists(harmful_file, normal_file)
        
        # Calculate statistics
        harmful_stats = self.stats_computer.calculate_stats(data['harmful'])
        normal_stats = self.stats_computer.calculate_stats(data['normal'])
        comparison = self.stats_computer.compare_datasets(harmful_stats, normal_stats)
        
        # Print results
        self._print_results(normal_stats, harmful_stats, comparison)
        
        # Create visualization
        fig = self.visualizer.create_comparison_bar_chart(
            normal_stats, harmful_stats, save_path
        )
        
        return {
            'normal_stats': normal_stats,
            'harmful_stats': harmful_stats,
            'comparison': comparison,
            'figure': fig
        }
    
    def _print_results(self, normal_stats: Dict, harmful_stats: Dict, comparison: Dict):
        """Print statistical results"""
        print(f"\nNormal Data Statistics:")
        print(f"  Mean: {normal_stats['mean']:.2f} ± {normal_stats['std']:.2f}")
        print(f"  Range: {normal_stats['min']:.2f} - {normal_stats['max']:.2f}")
        print(f"  Sample Size: {normal_stats['count']}")
        
        print(f"\nHarmful Data Statistics:")
        print(f"  Mean: {harmful_stats['mean']:.2f} ± {harmful_stats['std']:.2f}")
        print(f"  Range: {harmful_stats['min']:.2f} - {harmful_stats['max']:.2f}")
        print(f"  Sample Size: {harmful_stats['count']}")
        
        print(f"\nComparison Results:")
        print(f"  Mean Difference: {comparison['mean_difference']:.2f}")
        print(f"  Ratio (Harmful/Normal): {comparison['ratio']:.2f}")
        print(f"  Higher Perplexity: {comparison['higher_perplexity']}")

# =================================
#         CLI Interface
# =================================

def main():
    parser = ArgumentParser(description="Visualize saved perplexity data")
    parser.add_argument("--data_path", default="utils/data/llama2/perplexity", help="Path to saved perplexity data")
    parser.add_argument("--harmful_file", default="harmful_test.json", help="Harmful data filename")
    parser.add_argument("--normal_file", default="normal.json", help="Normal data filename")
    parser.add_argument("--save_path", default="utils/data/llama2/perplexity/figures", help="Path to save visualizations")
    
    args = parser.parse_args()
    
    # Run visualization pipeline
    pipeline = PerplexityVisualizationPipeline(args.data_path)
    results = pipeline.run_visualization(
        args.harmful_file, 
        args.normal_file, 
        args.save_path
    )
    
    print("\n=== VISUALIZATION COMPLETE ===")

if __name__ == "__main__":
    main()