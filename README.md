How the pipeline works:

1. Train the model using the training file---backdoor, obfuscated training
2. Get the activation via attn_store in utils
3. Compute crow to get the crow scores
4. Run safetynet file in analysis to get the scores for all detectors. 
5. Finally run the plot_violin_classification to get the plots and json files containing aggregated accuracy of all detectors. 







