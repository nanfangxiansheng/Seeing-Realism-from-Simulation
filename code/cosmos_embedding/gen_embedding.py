"""
This script uses the RobotWin dataset structure as an example to illustrate how to implement video embedding.
The final result is a JSON file containing the video embed code.

Assuming you are currently in the folder 'cosmos_embed_test/',
you simply need to provide the dataset path:

python gen_embedding.py \
  --dataset_root "/path/to/dataset" \
  --model_path "./new_model_weight" \
  --output_json "video_embeddings.json"\
  --num_frames 16
  
"""
import os
import json
import numpy as np
import torch
import re
import shutil
import decord
import argparse
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate video embeddings using Cosmos model.")
    
    # Path arguments
    parser.add_argument("--dataset_root", type=str, required=True, 
                        help="Root directory of the dataset.")
    parser.add_argument("--model_path", type=str, default="./model_weight", 
                        help="Path to the local model weights (default: ./model_weight).")
    parser.add_argument("--output_json", type=str, default="./video_embeddings.json", 
                        help="Path to save the output JSON (default: ./video_embeddings.json).")
    
    # Configuration arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu).")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample per video.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save JSON every N videos.")
    parser.add_argument("--max_videos_per_dir", type=int, default=300, help="Max videos to process per directory.")

    return parser.parse_args()

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, processor

def find_video_dirs(root):
    """
    Finds video directories based on specific dataset structure.
    Adjust the suffix check below if the dataset structure changes.
    """
    video_dirs = []
    target_suffix = os.path.join("aloha-agilex_randomized_500", "aloha-agilex_randomized_500", "video")#You need to modify the video path according to your actual file path.
    
    for dirpath, _, _ in os.walk(root):
        if dirpath.endswith(target_suffix):
            video_dirs.append(dirpath)
    return video_dirs

def embed_video(video_path, model, processor, device, num_frames):
    vr = decord.VideoReader(video_path)
    frame_ids = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    frames = vr.get_batch(frame_ids).float() / 255.0
    # Permute to (C, T, H, W) and add batch dim -> (1, C, T, H, W)
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0)

    with torch.no_grad():
        inputs = processor(videos=frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model.get_video_embeddings(**inputs)

    return output.visual_proj.cpu().squeeze(0)

def main():
    args = parse_args()
    
    # Set decord bridge
    decord.bridge.set_bridge("torch")

    # Load Model
    model, processor = load_model(args.model_path, args.device)

    # Load existing progress
    if os.path.exists(args.output_json):
        print(f"Loading existing results from {args.output_json}")
        try:
            with open(args.output_json, "r") as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing records.")
        except json.JSONDecodeError:
            print(f"[WARNING] JSON file corrupted. Backing up to {args.output_json}.bak and starting fresh.")
            shutil.copy(args.output_json, args.output_json + ".bak")
            all_results = {}
    else:
        print("No existing JSON found. Starting fresh.")
        all_results = {}

    # Find directories
    video_dirs = find_video_dirs(args.dataset_root)
    print(f"Found {len(video_dirs)} video directories.")

    total_processed_new = 0
    total_skipped = 0

    for video_dir in video_dirs:
        raw_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

        # Sort by episode number if possible, otherwise standard sort
        try:
            video_files = sorted(raw_files, key=lambda x: int(re.search(r'episode(\d+)', x).group(1)))
        except AttributeError:
            video_files = sorted(raw_files)
        
        # Limit number of videos per directory
        video_files = video_files[:args.max_videos_per_dir]

        print(f"\nEntering {video_dir}")
        print(f"Processing {len(video_files)} videos...")

        save_counter = 0

        for vf in tqdm(video_files, desc="Embedding", leave=True):
            video_path = os.path.join(video_dir, vf)

            # Skip if already processed
            if video_path in all_results:
                total_skipped += 1
                continue  

            try:
                emb = embed_video(video_path, model, processor, args.device, args.num_frames)

                all_results[video_path] = {
                    "embedding": emb.tolist(),
                    "dim": emb.numel()
                }
                
                save_counter += 1
                total_processed_new += 1

                # Periodic save
                if save_counter % args.save_interval == 0:
                    temp_file = args.output_json + ".tmp"
                    with open(temp_file, "w") as f:
                        json.dump(all_results, f)
                    os.replace(temp_file, args.output_json) 
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {video_path}: {e}")

        # Save after finishing a directory (if there were new entries)
        if save_counter > 0:
            temp_file = args.output_json + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(all_results, f)
            os.replace(temp_file, args.output_json)

    print(f"\n{'='*30}")
    print(f"Mission Complete")
    print(f"Total entries in JSON: {len(all_results)}")
    print(f"Newly processed this run: {total_processed_new}")
    print(f"Skipped (already existed): {total_skipped}")
    print(f"Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()