#Note: this script resizes head camera videos to 640x480 resolution and saves them to a new directory structure.
import json
import shutil
from pathlib import Path
import re
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
import os
import re
import subprocess
import os
root_path=Path("")
dst_root_path=Path("")
for first_level in root_path.iterdir():
    task_name=first_level.name
    video_dir=first_level/"videos"/"chunk-000"/"observation.images.head"
    for video_path in video_dir.iterdir():
        if video_path.suffix==".mp4":
            clip=VideoFileClip(str(video_path))
            resized_clip=clip.resized((640,480))
            video_name=video_path.name
            dst_video_path=dst_root_path/f"{task_name}"/"videos"/"chunk-000"/"observation.images.head"/f"{video_name}"
            dst_video_path.parent.mkdir(parents=True,exist_ok=True)
            resized_clip.write_videofile(str(dst_video_path))
            print(f"[OK] {video_path} -> {dst_video_path}")
