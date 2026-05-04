#Note:root_path and dst_root_path need to be set before running the script.
import json
import shutil
from pathlib import Path
import re
import os
root_path=Path("")
dst_root_path=Path("")
for first_level in root_path.iterdir():
    if (first_level/"robot_depth.mp4").is_file():
        with open(first_level/"control.json","r") as f:
            control_data=json.load(f)
            origin_video_path=control_data["video_path"]
            origin_video_name=str(origin_video_path).split("/")[-1]
            task_pattern = r'task<([^>]+)>'
            task_match = re.search(task_pattern, origin_video_name)
            if task_match:
                task_name = task_match.group(1)
            id_pattern = r'inputvideo(\d+)'
            id_match = re.search(id_pattern, origin_video_name)
            if id_match:
                video_id = id_match.group(1)
                video_id_padded = video_id.zfill(3)

            video_save_path=dst_root_path/f"{task_name}/videos/chunk-000/observation.images.head/episode_000{video_id_padded}.mp4"
            os.makedirs(video_save_path.parent,exist_ok=True)
            shutil.copy2(first_level/"robot_depth.mp4",video_save_path)
            print(f"[OK] {first_level/'robot_depth.mp4'} -> {video_save_path}")

            

