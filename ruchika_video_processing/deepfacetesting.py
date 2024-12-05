import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import gspread
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Mediapipe FaceMesh

with open("credentials.json", "r") as json_file:
    json_string = json.load(json_file)
SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/cloud-platform')
google_credentials = service_account.Credentials.from_service_account_info(json_string, scopes=SCOPES)
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "credentials.json"

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

class Update_data:
    def __init__(self):
        self.gc = gspread.authorize(google_credentials)  # Authorization of GoogleSheet
        Spreadsheet = self.gc.open_by_key('1PU2XJpmhHQ0D0RQAcJot2xBYg3wtIYbSlrx9ibySOIg')
        self.Data_C_Sheet = Spreadsheet.worksheet('Project Assignment')  # Source worksheet
        self.update_C_Sheet = Spreadsheet.worksheet('Facial Metrics')  # Destination worksheet
        self.datau_C_Sheet = Spreadsheet.worksheet('frame_data')  # Destination worksheet
        Dataa = self.update_C_Sheet.get_all_values()  # Get all values in the sheet
        self.dff= pd.DataFrame(Dataa[1:], columns=Dataa[0])
        Data = self.Data_C_Sheet.get_all_values()  # Get all values in the sheet
        self.df = pd.DataFrame(Data[1:], columns=Data[0])
        # folder_id=self.create_folder("r_image")
        row_count = 31621 																																																		
        for i in range(len(self.df)):
            if i>52:
                self.download_link(self.df.loc[i,'Link'])
                s_data = self.deepfacetesting('file_path2.mp4',i,row_count)
                df_da=s_data["dataframe"]
                df = df_da.fillna('')
                data_to_insert = df.values.tolist()
                Update_range = f"A{row_count}"  # update the range
                self.datau_C_Sheet.update(range_name=Update_range, values=data_to_insert)
                row_count=s_data["row_count"]
                    
                # self.update_C_Sheet.update([[self.df.loc[i, 'Link']]], f"A{row}")
                # self.update_C_Sheet.update([[s_data["average_positive"]]], f"B{row}")
                # self.update_C_Sheet.update([[s_data["average_negative"]]], f"C{row}")
                # self.update_C_Sheet.update([[s_data["average_neutral"]]], f"D{row}")
                # self.update_C_Sheet.update([[s_data["link_image"]]], f"E{row}")
                os.remove('file_path2.mp4')
             
    def create_folder(self,folder_name, parent_folder_id=None):
        """Create a folder in Google Drive and return its ID."""
        folder_metadata = {
            'name': folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            'parents': [parent_folder_id] if parent_folder_id else []
        }

        created_folder = drive_service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()

        print(f'Created Folder ID: {created_folder["id"]}')
        return created_folder["id"]

    def upload_image(self,file_path, folder_id):
        """Upload a single image to Google Drive and return its sharable link."""
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, mimetype='image/png')  # Adjust MIME type as needed
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        # Make the file sharable
        file_id = uploaded_file['id']
        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        # Get the sharable link
        file = drive_service.files().get(fileId=file_id, fields='webViewLink').execute()
        return file['webViewLink']
    def download_link(self, s3_url):
        response = requests.get(s3_url)
        if response.status_code == 200:
            with open("file_path2.mp4", "wb") as file:
                file.write(response.content)
                
    def calculate_emotion_scores(self,emotions,countt,tt):
       
        happy = emotions.get("happy", 0)
        sad = emotions.get("sad", 0)
        fear = emotions.get("fear", 0)
        disgust = emotions.get("disgust", 0)
        neutral = emotions.get("neutral", 0)
        # Total weight excluding neutral
        total_weight = happy + sad + fear + disgust

        
        return {
            "video_number":f"video_{tt}",
            "frame_number":countt,
             "positive": happy,
            "negative": sad+fear+disgust,
            "neutral": neutral
        }


    def deepfacetesting(self, video_path,tt,row_count):
        cap = cv2.VideoCapture(video_path)
        frame_results = []
        final_graph = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # DeepFace emotion detection
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                final_graph.append(self.calculate_emotion_scores(result['emotion'],frame_count,tt))
                
                
                
            except Exception as e:
                print(f"Error analyzing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()
        row_countt=frame_count+row_count
        # Save results to CSV for debugging
        # dff = pd.DataFrame(frame_results)
        # dff.to_csv('testingcsv.csv')

        # Visualize results
        dfgh = pd.DataFrame(final_graph)
        
        # categories = dfgh.columns
        # values = [dfgh[category] for category in categories]
        # category_colors = {
        #     "positive": "green",
        #     "negative": "red",
        #     "neutral": "yellow"
        # }

        # # Extract the colors in the order of categories
        # colors = [category_colors[category] for category in categories]

        # plt.figure(figsize=(14, 7))
        # plt.stackplot(dfgh.index, *values, labels=categories, colors=colors, alpha=0.8)
        # # Add titles and labels
        # plt.title('Sentiment Analysis Over Time (Stacked Area Chart)', fontsize=16)
        # plt.xlabel('Frame Number', fontsize=12)
        # plt.ylabel('Emotion Value', fontsize=12)
        # plt.legend(loc='upper left', title='Sentiment')
        # plt.tight_layout()
        # # Show the plot
        # plt.savefig(f"pathimage_{tt}.png", format='png', dpi=300)
        # link_image=self.upload_image(f"pathimage_{tt}.png",folder_id)
        # plt.close()  
        # # Calculate averages
        # summary={'average_positive':dfgh["positive"].mean(),
        #          'average_negative':dfgh["negative"].mean(),
        #          'average_neutral':dfgh["neutral"].mean(),
        #          "link_image":link_image
        #          }
        #os.remove(f"pathimage_{tt}.png")
        summary={
             "dataframe":dfgh,
             "row_count":row_countt
                  }
        return summary

Update_data()