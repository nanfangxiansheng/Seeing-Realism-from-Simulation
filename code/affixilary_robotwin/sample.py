import os
import shutil
import re

def copy_and_rename_videos(
    paths_txt="paths.txt",#paths.txt refers to the txt file containing the video paths selected by core set selection
    output_dir=""
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/instructions",exist_ok=True)

    with open(paths_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    total_counter = 0
    for video_path in lines:
        m = re.search(r"episode(\d+)\.mp4$", video_path)
        episode_id = m.group(1)
        parts = video_path.split(os.sep)
        dataset_idx = parts.index("dataset")
        task_name = parts[dataset_idx + 1]
        instructions_path=video_path.split("video")[0]
        instructions_path=instructions_path+"instructions"+f"/episode{episode_id}.json"
        dest_instruction_path=output_dir+"/instructions"+f"/totalcounter{total_counter}.json"
        new_name = (
            f"totalcounter({total_counter})_"
            f"task<{task_name}>_"
            f"inputvideo{episode_id}.mp4"
        )
        dst_path = os.path.join(output_dir, new_name)
        shutil.copy2(video_path, dst_path)#careful to use copy2 to preserve metadata
        shutil.copy2(instructions_path,dest_instruction_path)

        print(f"[OK] {video_path} -> {dst_path}")
        total_counter += 1

if __name__ == "__main__":
    copy_and_rename_videos(
        paths_txt="",
        output_dir=""
    )
