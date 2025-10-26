from utils import *
from src.configs.safetynet_config import SafetyNetConfig
from utils.safetynet.vae_ae_train import Attention_DataProcessing, Train, Test, Detector_Stats


import plotly.graph_objects as go



class Visualization:
    
    @staticmethod
    def data_processing_for_crow(
        other_layer_idx,
        vanilla_path = "utils/data/llama2/ae_vae/vanilla/cosine_analysis.json", 
        harmful_path = "utils/data/llama2/ae_vae/backdoored/cosine_analysis.json"):
        
        with open(vanilla_path, "r") as f:
            vanilla_data = json.load(f)
        
        with open(harmful_path, "r") as f_:
            backdoor_data = json.load(f_)
        
        if other_layer_idx == 'prev':
            layer_idx = 0
        elif other_layer_idx == "next":
            layer_idx = 1
            
            
        '''
        As the two layers pair values are there, so having [0] will give the first pair and [1] the second pair
        '''
        mean_harmful_vanilla = np.mean(np.array(vanilla_data["harmful"][layer_idx]))
        mean_harmful_backdoor = np.mean(np.array(backdoor_data["harmful"][layer_idx]))
        
        vanilla_data_stats = [i - float(mean_harmful_vanilla) for i in vanilla_data["normal"][layer_idx]]
        # Fix the list slicing - use int() for indices
        split_idx = int(len(vanilla_data_stats) * 0.8)
        vanilla_data_stats_train = vanilla_data_stats[:split_idx]
        vanilla_data_stats_val = vanilla_data_stats[split_idx:]
        backdoor_data_stats = [i - float(mean_harmful_backdoor) for i in backdoor_data["normal"][layer_idx]]        
        
        
        # Return as a dictionary to match expected format
        return {
            "normal_losses": vanilla_data_stats_train,
            "val_losses": vanilla_data_stats_val, 
            "harmful_losses": backdoor_data_stats
        }
        
        
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
    def plot_detectors_comparison(model_name, 
                                  detector_types, 
                                  other_layer_idx, 
                                  current_layer_idx, 
                                  save_path, 
                                  config: SafetyNetConfig, 
                                  model_type
                                  ):
        """Compare different detector types (AE, VAE, PCA) at a specific layer with normalized losses"""
        
        fig = go.Figure()
        
        results = {}
        
        for i, detector_type in enumerate(detector_types):
            
            if detector_type == "crow"  \
                or detector_type == "obfuscated_sim_crow" \
                    or detector_type == "obfuscated_ae_crow":
                # Get data as dictionary
                data = Visualization.data_processing_for_crow(
                    other_layer_idx=other_layer_idx,
                    harmful_path = f"utils/data/llama2/ae_vae/{model_type}/cosine_analysis.json"
                    )
                normal_losses = np.array(data["normal_losses"])
                harmful_losses = np.array(data["harmful_losses"])
                val_losses = np.array(data["val_losses"])
                
            else:
                data_path = f"utils/data/{model_name}/{detector_type}_loss/layer_{current_layer_idx}_{detector_type}_loss.json"
                print(data_path)
                with open(data_path, "r") as f:
                    data = json.load(f)
                    
                
                # Extract losses
                normal_losses = np.array(data["normal_losses"])
                harmful_losses = np.array(data["harmful_losses"])
                val_losses = np.array(data["val_losses"])
                
                print(data)
                # print(harmful_losses)
                # print(val_losses)
                
            
            val_mean = np.mean(val_losses)
            val_std = np.std(val_losses)
            threshold_upper = val_mean + 2 * val_std
            threshold_lower = val_mean - 2 * val_std
            

            # Predictions (1 = anomaly/harmful, 0 = normal)
            train_pred = ((normal_losses < threshold_lower) | (normal_losses > threshold_upper)).astype(int)
            harmful_pred = ((harmful_losses < threshold_lower) | (harmful_losses > threshold_upper)).astype(int)

            # Labels
            train_labels = np.zeros(len(normal_losses))
            harmful_labels = np.ones(len(harmful_losses))

            # Combine everything
            all_pred = np.concatenate([train_pred, harmful_pred])
            all_labels = np.concatenate([train_labels, harmful_labels])
            all_scores = np.concatenate([normal_losses, harmful_losses])

            # AUROC: Check if scores need to be inverted
            # If lower scores = more anomalous, negate them
            # try:
            stats = Detector_Stats()
            detector_results = stats.compute_comprehensive_metrics(normal_losses, val_losses, harmful_losses)
            detector_results['confusion_matrix_overall'] = detector_results['confusion_matrix_overall'].tolist()
            results[detector_type] = detector_results
            '''
            print(all_labels)
            print(all_scores)
            auroc = roc_auc_score(all_labels, all_scores)
            if auroc < 0.5:  # Scores are inverted
                auroc = roc_auc_score(all_labels, -all_scores)
            # except:
            #     auroc = 0.5

            # Overall metrics
            overall_accuracy = accuracy_score(all_labels, all_pred)
            overall_precision = precision_score(all_labels, all_pred, zero_division=0)
            overall_recall = recall_score(all_labels, all_pred, zero_division=0)
            overall_f1 = f1_score(all_labels, all_pred, zero_division=0)

            # Per-class metrics
            train_accuracy = np.mean(train_pred == train_labels)
            harmful_accuracy = np.mean(harmful_pred == harmful_labels)
            harmful_precision = precision_score(harmful_labels, harmful_pred, zero_division=0)
            harmful_recall = recall_score(harmful_labels, harmful_pred, zero_division=0)
            harmful_f1 = f1_score(harmful_labels, harmful_pred, zero_division=0)

            results[detector_type] = {
                "auroc": float(auroc),
                "overall_accuracy": float(overall_accuracy),
                "overall_precision": float(overall_precision),
                "overall_recall": float(overall_recall),
                "overall_f1": float(overall_f1),
                "train_accuracy": float(train_accuracy),
                "harmful_accuracy": float(harmful_accuracy),
                "harmful_precision": float(harmful_precision),
                "harmful_recall": float(harmful_recall),
                "harmful_f1": float(harmful_f1),
                "threshold_lower": float(threshold_lower),
                "threshold_upper": float(threshold_upper)
            }
            '''
            
            
            # Normalize to 0-1 range using min-max scaling across all loss types
            all_losses = np.concatenate([normal_losses, harmful_losses, val_losses])
            min_loss = np.min(all_losses)
            max_loss = np.max(all_losses)
            loss_range = max_loss - min_loss
            
            print(f"\n NORMAL LOSSEs \n")
            print(normal_losses)
            
            
            # Avoid division by zero
            if loss_range == 0:
                loss_range = 1
            
            # Normalize each loss type
            normal_norm = (normal_losses - min_loss) / loss_range
            harmful_norm = (harmful_losses - min_loss) / loss_range
            val_norm = (val_losses - min_loss) / loss_range
            
            
            detector = detector_type.split("_")[-1]
            
            if detector == "crow":
                if other_layer_idx == "prev":
                    x_pos = f"CROW {current_layer_idx-1}-{current_layer_idx}"
                elif other_layer_idx == "next":
                    x_pos = f"CROW {current_layer_idx}-{current_layer_idx+1}"
            else:
                x_pos = detector.upper()
            
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
                
            
            print(f"CURRENTLY PROCESSING {detector_type}")


            
        pprint(results)
        
        # Layout with improved styling
        fig.update_layout(
            title=dict(
                text=f'{model_type.upper()} {config.model_name} Detector Comparison at Layer {current_layer_idx}',#<br><sub>Losses Normalized to [0,1] Range</sub>',
                x=0.5, y=0.96, xanchor='center', yanchor='top',
                font=dict(family="Times New Roman", size=12, color="black")
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
        
        if other_layer_idx == "prev":
            fig.write_image(f"{save_path}_{model_type}_detectors_comparison_layer_{current_layer_idx-1}_{current_layer_idx}.pdf", 
                            height=300, width=500, scale=3)
            

            # At the end of the method, before return:
            accuracy_path = f"{save_path}_{model_type}_accuracy_layer_{current_layer_idx-1}_{current_layer_idx}.json"
        
        elif other_layer_idx == "next":
            fig.write_image(f"{save_path}_{model_type}_detectors_comparison_layer_{current_layer_idx}_{current_layer_idx+1}.pdf", 
                            height=300, width=500, scale=3)
            

            # At the end of the method, before return:
            accuracy_path = f"{save_path}_{model_type}_accuracy_layer_{current_layer_idx}_{current_layer_idx+1}.json"
            
        
        def numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: numpy_to_python(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            return obj

        # Convert entire results dictionary
        results = numpy_to_python(results)

        if 'confusion_matrix_overall' in results:
            cm = results['confusion_matrix_overall']
            if isinstance(cm, np.ndarray):
                results['confusion_matrix_overall'] = cm.tolist()
            elif isinstance(cm, list):
                results['confusion_matrix_overall'] = [[int(x) for x in row] for row in cm]


        # Now save
        if os.path.exists(accuracy_path):
            with open(accuracy_path, 'r') as f:
                existing_results = json.load(f)
            existing_results.update(results)
            results = existing_results
        
        with open(accuracy_path, 'w') as f:
            json.dump(results, f, indent=2)
                
        
        return fig

# Updated main section:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-layer Attention Analysis')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument("--other_layer_idx", type=str, required=True, help="crow should be taken for previous and current layer or next and current layers? give 'prev' or 'next' as argument ")
    args = parser.parse_args()
    config = SafetyNetConfig(args.model_name)
    
    if args.model_name == 'qwen':
        current_layer_idx=21
    elif args.model_name == 'mistral':
        current_layer_idx = 12
    elif args.model_name == 'llama3':
        current_layer_idx = 13
    elif args.model_name == 'llama2':
        current_layer_idx = 15
    elif args.model_name == 'gemma':
        current_layer_idx = 18
    
    # save_path = f"{config.output_dir}/{args.model_name}_all_layers_{args.model_type}"
    save_path = f"{config.output_dir}/{args.model_name}"
    
    viz = Visualization()
    # viz.plot_all_layers_violin(args.model_name, args.model_type, save_path, config=config)
    viz.plot_detectors_comparison(args.model_name, 
                                #   ['ae', 'vae', 'pca', 'mahalanobis', 'beatrix', f'crow'], 
                                # ['ae', 'pca', 'mahalanobis', 'beatrix', f'crow'], 
                                #   ['obfuscated_sim_ae', 
                                #    'obfuscated_sim_vae', 
                                #    'obfuscated_sim_pca', 
                                #    'obfuscated_sim_mahalanobis', 
                                #    'obfuscated_sim_beatrix',
                                #    'obfuscated_sim_crow'], 
                                  ['obfuscated_ae_ae', 
                                #    'obfuscated_ae_vae', 
                                   'obfuscated_ae_pca', 
                                   'obfuscated_ae_mahalanobis', 
                                   'obfuscated_ae_beatrix',
                                   'obfuscated_ae_crow'],
                                  current_layer_idx = current_layer_idx, 
                                  other_layer_idx = args.other_layer_idx,
                                  save_path = save_path, 
                                  config = config,
                                  model_type = args.model_type,
                                  )
    
    
# python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx prev --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx next --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx prev --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx next --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx prev --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx next --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx prev --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx next --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx prev --model_name qwen && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_sim --other_layer_idx next --model_name qwen
# python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx prev --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx next --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx prev --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx next --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx prev --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx next --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx prev --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx next --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx prev --model_name qwen && python -m utils.visualisation.plot_violin_classification --model_type backdoored --other_layer_idx next --model_name qwen
# python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx prev --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx next --model_name gemma && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx prev --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx next --model_name mistral && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx prev --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx next --model_name llama2 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx prev --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx next --model_name llama3 && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx prev --model_name qwen && python -m utils.visualisation.plot_violin_classification --model_type obfuscated_ae --other_layer_idx next --model_name qwen
