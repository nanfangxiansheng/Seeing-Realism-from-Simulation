#Note: Please set the 'root_path' and 'dst_dir' variables before running the script.
import json
import shutil
from pathlib import Path
import re
root_path=Path("")
dst_dir=Path("")
total_counter=0
record_data=[]
for first_level in root_path.iterdir():
    if (first_level/"videos").is_dir():
        task_name=str(first_level).split("/")[-1]
        for video_path in (first_level/"videos/chunk-000/observation.images.head").iterdir():
            if video_path.suffix==".mp4":
                video_name=str(video_path).split("/")[-1]
                match = re.search(r'episode_(\d+)', video_name)
                if match:
                    number = int(match.group(1))
                dst_name=f"totalcounter({total_counter+1})_task<{task_name}>_inputvideo{number}.mp4"
                shutil.copy2(video_path,dst_dir/dst_name)
                total_counter+=1
                print(f"[OK] {video_path} -> {dst_dir/dst_name}")
                record_data.append({"original_video_path":str(video_path),"dst_video_path":str(dst_dir/dst_name)})


with open(dst_dir/"record.json","w") as f:
    json.dump(record_data,f,indent=4)




