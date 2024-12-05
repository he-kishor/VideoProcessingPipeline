
import sys
import json
import pandas as pd
from google.oauth2 import service_account
from s3_downloadfile import Download_video
from check_video.video_layout import Video_modification
from moviepy.editor import ImageClip, concatenate_videoclips, VideoFileClip
import os
import subprocess
import gspread
json_file="googleauthconnection.json"

with open(json_file, "r") as json_file:
    json_string = json.load(json_file)
SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/cloud-platform')
google_credentials = service_account.Credentials.from_service_account_info(json_string, scopes=SCOPES)

class GoogleSheetcommunicate:
    def __init__(self):
        self.gc = gspread.authorize(google_credentials)  # Authorization Of GoogleSheer
        Spreadsheet = self.gc.open_by_key('1IdlmnC3AiLNBoecMEyLGQGvuGOg3XSRG82cgQ0GXuEA')
        self.Video_processSheet= Spreadsheet.worksheet('video_concat') 
        data_vi=self.Video_processSheet.get_all_values()
        self.df_v = pd.DataFrame(data=data_vi[1:], columns=data_vi[0])
        self.v_obj=Video_modification
        self.js_fun={
            "image_inside_downvideo":self.v_obj.combine_video_image_image_inside_downvideo,
            "video_right_insideimage":self.v_obj.combine_video_image_video_right_insideimage,
            "fade_in_out_image":self.v_obj.fade_in_out_image,
            "overlay_image_bottom_left":self.v_obj.overlay_image_bottom_left,
            "overlay_transparent_image_bottom":self.v_obj.overlay_transparent_image_bottom,
            "slide_in_image_from_right":self.v_obj.slide_in_image_from_right,
            "plit_screen_image_right":self.v_obj.split_screen_image_right,
            "watermark_image_top_right":self.v_obj.watermark_image_top_right,
        }
        self.lis_vobj = []
    def call_function_with_args(self,func_name, *args, **kwargs):
        if func_name in self.js_fun:
            # Call the function stored in the dictionary with arguments
            final_clip=self.js_fun[func_name](*args, **kwargs)
            return final_clip
        else:
            print(f"Function {func_name} not found.")
            return None
    #in nested loop
     #call function using url download the vide then converted it into the video_clip objects
     #call the function cut the video which return the cuttet video object and return this
     #call the function that attacht the image and make the design and return the final design video object
     #append this return video object into the list 
     
    def set_duration(self):
        t=Download_video()
        for i in range( len(self.df_v)):
            video_clicp=t.video_from_url(self.df_v.loc[i,'url'])
            video_clip=self.v_obj.cut_video(video_clicp,self.df_v.loc[i,'Start_time'],self.df_v.loc[i,'end_time'])
            image_path=f"../check_video/{self.df_v.loc[i,'template_name']}"
            image_clip= ImageClip(image_path)
            #image_clip.save_frame("image.png")
            final_clip=self.call_function_with_args(self.df_v.loc[i,'position'],video_clip,image_clip)
            if final_clip is not None:
                #final_clip.write_videofile(f'testing_video{i}.mp4', codec='libx264', fps=video_clip.fps)
                self.lis_vobj.append(final_clip)
      
        main_clip = concatenate_videoclips(self.lis_vobj,method="compose")
        main_clip.write_videofile("main_output.mp4", threads = 8, fps=24)
        self.concatenate_videos(self.lis_vobj,"final_output_video.mp4")
        
    def concatenate_videos(self,video_clips, output_path):
         # Create a temporary file to store the list of video files
        with open("video_list.txt", "w") as f:
            for video in video_clips:
                f.write(f"file '{video.filename}'\n")

        # Temporary output file for concatenation
        temp_output = "temp_output.mp4"

        # Use ffmpeg to concatenate the videos with resizing
        concat_cmd = [
            'ffmpeg',
            '-y',  # Overwrite the output file if it exists
            '-f', 'concat',
            '-safe', '0',
            '-i', 'video_list.txt',
            '-vf', 'scale=240:380',  # Resize to a common resolution
            '-c', 'libx264',  # Use x264 codec for video
            '-c:a', 'aac',  # Use aac codec for audio
            temp_output
        ]

        subprocess.run(concat_cmd, check=True)

        # Move the temporary output to the final output path
        os.rename(temp_output, output_path)

        # Clean up the temporary files
        os.remove("video_list.txt")
                    
                
            
        
d=GoogleSheetcommunicate()
d.set_duration()