import gspread
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
with open("credentials.json", "r") as json_file:
    json_string = json.load(json_file)
SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/cloud-platform')
google_credentials = service_account.Credentials.from_service_account_info(json_string, scopes=SCOPES)
class Update_data:
    def __init__(self):
        self.gc = gspread.authorize(google_credentials)  # Authorization of GoogleSheet
        Spreadsheet = self.gc.open_by_key('1PU2XJpmhHQ0D0RQAcJot2xBYg3wtIYbSlrx9ibySOIg')
        self.datau_C_Sheet = Spreadsheet.worksheet('frame_data')  # Destination worksheet
        Dataa = self.datau_C_Sheet.get_all_values()  # Get all values in the sheet
        self.dff= pd.DataFrame(Dataa[1:], columns=Dataa[0])
        self.dff['Frame_number']=self.dff['Frame_number'].astype('int')
        self.dff['PositiveScore']=self.dff['PositiveScore'].astype('double')
        self.dff['NegativeScore']=self.dff['NegativeScore'].astype('double')
        self.dff['NeuttralScore']=self.dff['NeuttralScore'].astype('double')
        # Group by Frame_number and calculate the average for each score column
        average_scores = self.dff.groupby('Frame_number')[['PositiveScore', 'NegativeScore', 'NeuttralScore']].mean().reset_index()

        # Rename the columns for clarity
        average_scores.rename(columns={
            'PositiveScore': 'Avg_PositiveScore',
            'NegativeScore': 'Avg_NegativeScore',
            'NeuttralScore': 'Avg_NeutralScore'
        }, inplace=True)

        df=average_scores.iloc[:400,:]
        df.set_index('Frame_number', inplace=True)

        # Prepare data for the stacked area chart
        categories = ["Avg_PositiveScore", "Avg_NegativeScore", "Avg_NeutralScore"]
        values = [df[category] for category in categories]

        # Define colors for each category
        category_colors = {
            "Avg_PositiveScore": "green",
            "Avg_NegativeScore": "red",
            "Avg_NeutralScore": "yellow"
        }

        # Extract colors in the order of categories
        colors = [category_colors[category] for category in categories]

        # Create the stacked area chart
        plt.figure(figsize=(14, 7))
        plt.stackplot(df.index, *values, labels=categories, colors=colors, alpha=0.8)

        # Add titles and labels
        plt.title('Sentiment Analysis Over Time (Stacked Area Chart)', fontsize=16)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Emotion Value', fontsize=12)
        plt.legend(loc='upper left', title='Sentiment')
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("pathimage.png", format='png', dpi=300)

Update_data()