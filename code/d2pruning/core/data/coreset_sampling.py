""""
You can now run the script from the command line by specifying the paths:

python d2pruning/core/data/coreset_sampling.py.py \
  --input_path "./core/data/merged_dataset_input.json" \
  --output_path "./core/data/coreset.json" \
  --subset_size 2000 \
  --stratas 50 \
  --budget_mode confidence
"""


import json
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from Coreset import CoresetSelection 

# --- Mock Argument Class (Refactored) ---
class SamplingConfig:
    """
    A wrapper class to mimic the configuration object expected by CoresetSelection.
    """
    def __init__(self, coreset_key='score', stratas=50, budget_mode='confidence', sampling_mode='kcenter'):
        # Core parameters
        self.coreset_key = coreset_key
        self.stratas = stratas
        self.budget_mode = budget_mode
        self.sampling_mode = sampling_mode
        
        # --- Placeholder parameters for Coreset.py compatibility ---
        self.graph_score = False      # Prevents attribute errors
        self.aucpr = False            # Prevents missing aucpr error
        self.gamma = None             # Graph sampling parameter
        self.n_neighbor = 10          # Graph sampling neighbors
        self.graph_mode = None        # Placeholder
        self.precomputed_dists = None # Placeholder
        self.precomputed_neighbors = None # Placeholder

def parse_args():
    parser = argparse.ArgumentParser(description="Perform Stratified Coreset Selection on Video Dataset.")
    
    # I/O Paths
    parser.add_argument("--input_path", type=str, required=True, 
                        help="Path to the merged dataset JSON file.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the resulting subset JSON.")
    
    # Sampling Parameters
    parser.add_argument("--subset_size", type=int, default=2000, 
                        help="Target number of samples to select.")
    parser.add_argument("--stratas", type=int, default=50, 
                        help="Number of intervals (bins) to split the difficulty score into.")
    parser.add_argument("--budget_mode", type=str, default='confidence', choices=['confidence', 'uniform'],
                        help="Budget allocation strategy: 'confidence' (more budget for hard samples) or 'uniform'.")
    parser.add_argument("--sampling_mode", type=str, default='kcenter', choices=['kcenter', 'random'],
                        help="Sampling strategy within strata: 'kcenter' (geometry-based) or 'random'.")
    
    return parser.parse_args()

def load_and_process_json(json_path):
    """
    Reads the JSON and flattens hierarchy into Tensors/Numpy arrays.
    """
    print(f"Loading data from: {json_path} ...")
    
    # Assumes JSON structure: { "Task_A": [ {video_info}, ... ], "Task_B": ... }
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_features = []
    all_scores = []      # Corresponds to difficulty/loss
    all_targets = []     # Corresponds to task_id (e.g., 0~31)
    
    # Metadata list to retrieve video ID or path via index later
    meta_info_list = [] 

    # Iterate through tasks (sorted to ensure consistent ID mapping)
    task_names = sorted(list(raw_data.keys()))
    task_to_id = {name: i for i, name in enumerate(task_names)}

    for task_name, videos in raw_data.items():
        task_id = task_to_id[task_name]
        
        for vid in videos:
            # 1. Extract Embedding (Assuming list format)
            emb = vid.get('embedding') 
            # 2. Extract Difficulty Score (Loss)
            score = vid.get('difficulty') 
            
            # Simple validation
            if emb is None or score is None:
                continue

            all_features.append(emb)
            all_scores.append(score)
            all_targets.append(task_id)
            
            # Save metadata for later retrieval
            meta_info_list.append({
                "task": task_name,
                "video_id": vid.get('id', 'unknown'),
                "path": vid.get('path', ''),
                "original_data": vid # Store the whole object reference
            })

    # Convert to Numpy/Tensor
    # Features: (N_samples, D_dim)
    features_np = np.array(all_features, dtype=np.float32)
    
    # Scores: (N_samples,)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    
    # Targets: (N_samples,)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    # Construct confidence (Optional)
    # Assuming score is Loss, Confidence ~= 1 / (1 + Loss) or simple normalization
    # This simulates confidence; relevant if budget_mode='confidence'
    max_loss = torch.max(scores_tensor)
    confidence_tensor = 1.0 - (scores_tensor / (max_loss + 1e-6))

    data_score = {
        'score': scores_tensor,      # The difficulty metric
        'targets': targets_tensor,   # The task IDs
        'confidence': confidence_tensor 
    }

    print(f"Data loaded. Total samples: {len(meta_info_list)}, Feature dim: {features_np.shape[1]}")
    return data_score, features_np, meta_info_list

def save_selection(selected_indices, meta_info_list, output_path, data_score=None):
    """
    Saves the selected samples to a new JSON file, appending the sampling score.
    """
    selected_data = []
    
    # Get all scores tensor if provided
    all_scores = data_score['score'] if data_score else None
    
    print("Preparing data for export...")
    for idx in selected_indices:
        # Note: idx might be a tensor, convert to int
        i = int(idx)
        
        # 1. Copy original data (use copy to avoid modifying memory references)
        item = meta_info_list[i]['original_data'].copy()
        
        # 2. Inject score (if available)
        if all_scores is not None:
            # .item() converts tensor to Python float for JSON serialization
            score_value = all_scores[i].item()
            
            # Save as 'sampling_score' or overwrite 'difficulty' based on preference
            item['sampling_score'] = round(score_value, 6) 
            
        selected_data.append(item)
    
    print(f"Saving {len(selected_data)} selected samples to {output_path} ...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=4)

# --- Main Execution Logic ---

def main():
    # 1. Parse Arguments
    args_cli = parse_args()
    
    # 2. Load Data
    if not os.path.exists(args_cli.input_path):
        print(f"Error: Input file not found: {args_cli.input_path}")
        return

    data_score, features, meta_info = load_and_process_json(args_cli.input_path)
    
    # 3. Sampling Setup
    print(f"Total samples: {len(meta_info)} -> Target subset size: {args_cli.subset_size}")
    
    # Configure the sampling strategy object
    # Stratified + Confidence Budget + K-Center
    config = SamplingConfig(
        coreset_key='score',             # Use difficulty (loss) for stratification
        stratas=args_cli.stratas,        # Number of bins
        budget_mode=args_cli.budget_mode,# Dynamic budget allocation based on difficulty
        sampling_mode=args_cli.sampling_mode # Selection strategy within bins
    )
    
    print(f"Starting Stratified Sampling...")
    print(f"Strategy: Stratas={config.stratas}, Budget={config.budget_mode}, Sampling={config.sampling_mode}")

    if CoresetSelection is None:
        print("Error: CoresetSelection module is missing. Cannot proceed.")
        return

    # 4. Execute Sampling
    selected_indices, _ = CoresetSelection.stratified_sampling(
        data_score, 
        args_cli.subset_size, 
        config, 
        data_embeds=features # Pass Embeddings for K-Center calculation
    )

    # 5. Save Results
    save_selection(selected_indices, meta_info, args_cli.output_path, data_score=data_score)
    print(f"Mission Complete! Results saved to: {args_cli.output_path}")

if __name__ == "__main__":
    main()


