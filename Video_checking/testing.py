from moviepy.editor import VideoFileClip, ImageClip, clips_array, CompositeVideoClip, concatenate_videoclips
import gspread
import json
import pandas as pd
from google.oauth2 import service_account
from s3_downloadfile import Download_video
import ffmpeg
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
        self.moviepyconcating()
    def moviepyconcating(self):
        vd=Download_video()
        vd.video_from_url(self.df_v.loc[0,'url'],"t1.mp4")
        v1clip=VideoFileClip('t1.mp4')
        vd.video_from_url(self.df_v.loc[2,'url'],"t2.mp4")
        v2clip=VideoFileClip('t2.mp4')
        
        v3clip=VideoFileClip('testing24.mp4')
        v4clip=VideoFileClip('testing25.mp4')
        v6clip=VideoFileClip('jasimtesting1.mp4')
        v7clip=VideoFileClip('tanmaytestingnn2.mp4')
        final_clip=concatenate_videoclips([v1clip,v2clip])
        final_clip.write_videofile("main_output.mp4")
        
    def concating_ffmpeg(self):
        video_files = ['hemanttesting1.mp4', 'jasimtestingnn2.mp4',]
        output_file = 'output.mp4'
        input_streams = [ffmpeg.input(video) for video in video_files]
        ffmpeg.concat(*input_streams).output(output_file).run()


GoogleSheetcommunicate()