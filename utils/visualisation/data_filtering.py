from utils import *
from src.configs.model_configs import AnalysisConfig


def visualize_prompt_distribution(model_name: str, 
                                  config: AnalysisConfig, 
                                  dataset_type:str, 
                                  prompt_lengths:int, 
                                  best_start: int, 
                                  best_end: int, 
                                save_path=None, title=None, show_plot=True) -> None:
    """Create bar chart instead of histogram for prompt length distribution"""
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

    # Add all bars
    colors = ['#4DB6AC' if best_start <= x < best_end else 'lightblue' for x in bin_centers]

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=colors,
        name='Prompt Length Distribution',
        text=[f'{count}' if count > 0 else '' for count in counts],
        textposition='outside'
    ))

    # Add vertical lines for range boundaries
    fig.add_vline(x=best_start, line_dash="dash", line_color="#00695C", 
                    annotation_text=f"Start: {best_start}")
    fig.add_vline(x=best_end, line_dash="dash", line_color="#00695C", 
                    annotation_text=f"End: {best_end}")

    # Layout
    default_title = (f"{config.model_name} on {dataset_type} dataset [{best_start}, {best_end}) " +
                    f"({best_count}/{len(prompt_lengths)} samples, {percentage:.1f}%)")

    fig.update_layout(
        title={'text': title or default_title, 'x': 0.5, 'font': {'size': 18}},
        xaxis_title="Prompt Length (tokens)",
        yaxis_title="Frequency",
        template='plotly_white',
        plot_bgcolor='#FFFEF7',
        paper_bgcolor='white',
        width=800,
        height=500,
        showlegend=False,
        bargap=0.1  # Small gap between bars
    )

    # Save HTML only (no image export issues)
    fig.write_image(save_path, width=700, height=500, scale=2)

    return fig