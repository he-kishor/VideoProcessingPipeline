import requests
import time 
from moviepy.editor import VideoFileClip
import os
class Download_video:
    def __init__(self):
        self.save_path="temp_video.mp4"

        
    def download_video(self,url, save_path):
    # Send a GET request to download the video
        response = requests.get(url, stream=True)
        time.sleep(2)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)  # Directly write the entire content at once
            print(f"Video downloaded successfully: {save_path}")
        else:
            print(f"Failed to download video. Status code: {response.status_code}")


    def video_from_url(self,url,path):
        # Define a temporary file path to save the video
        temp_video_path = "temp_video.mp4"
        self.download_video(url, path)
        time.sleep(2)  # Add a small delay to ensure the file is fully closed
        while not os.path.exists(temp_video_path):
            time.sleep(1)

        video_clip = VideoFileClip(temp_video_path)
        
        return video_clip

  