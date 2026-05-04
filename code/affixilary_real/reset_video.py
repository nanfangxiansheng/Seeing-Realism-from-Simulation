#Note: This script resizes videos in a specified directory to 960x720 resolution
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

def print_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"pixels: {width}x{height} ")
    cap.release()


    
dest_dir1 = Path("./")
write_dir2=Path("./")
if dest_dir1.exists() and dest_dir1.is_dir():
    video_files = sorted([
    f for f in dest_dir1.iterdir() 
    if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    ])
    for video_file in video_files:
        clip=VideoFileClip(video_file)
        resized_clip = clip.resized((960, 720))
        video_name=str(video_file).split("/")
        video_name=video_name[-1]

        resized_clip.write_videofile(f"./reset/{video_name}")
        print_video_resolution(video_file)
        clip.close()
        resized_clip.close()
    