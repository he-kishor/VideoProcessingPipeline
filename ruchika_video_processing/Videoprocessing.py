import cv2.data
import gspread
from google.oauth2.service_account import Credentials
import os
import time
import requests
from moviepy.editor import VideoFileClip
from pydub import AudioSegment, silence
import noisereduce as nr
from scipy.signal import find_peaks
from scipy.io import wavfile
import whisper
import re
import nltk
import psutil
import langid
import tracemalloc
import parselmouth
import cv2
import numpy as np
import threading
import queue
import pandas as pd


# queue intializing 
queue1 = queue.Queue()
queue2 = queue.Queue()
queue3 = queue.Queue()
queue4 = queue.Queue()
queue5 = queue.Queue()
queue6 = queue.Queue()
queue7 = queue.Queue()

logical_cores = os.cpu_count()
#print(logical_cores,"logical cpu count")

#Download CPU Pronouncing Dictionary if not already download
nltk.download('cmudict')
nltk.download('punkt')
cmu_dict = nltk.corpus.cmudict.dict()
def authenticate_gspread(credentials_file):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
    client = gspread.authorize(creds)
    return client

#Intialize face cascade classifier
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def download_video(q_in, q_out):
    #ruchika logic
    #using the url file 
    #download the file using url into the system 
    while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
        try:
            video_url=item['video_url']
            idx=item['idx']
            file_name = f"video_{idx+1}.mp4"
            response = requests.get(video_url, timeout=30)
            with open(file_name, 'wb') as video_file:
                video_file.write(response.content)
            data={
                "video_url":item['video_url'],
                'file_name':file_name,
                'idx':idx,
                'analysis_sheet':item['analysis_sheet']
            }
           
            q_out.put(data)
            time.sleep(1)
            
        except Exception as e:
            pass
            #print(f"Error downloading video {video_url}: {e}")
            
def extract_and_preprocess_audio(q_in,q_out):
    while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
    
        try:
            #ruchika logic
            #convert the video file intot video object so we can intract with object using moviepy 
            #convert audio into the audio file 
            #create audio.wav file
            video_file=item['file_name']
            
            video = VideoFileClip(video_file)
            audio = video.audio
            audio_file = f"audio.wav"
            audio.write_audiofile(audio_file, codec='pcm_s16le')

            # Preprocess the audio using pydub (convert to mono, reduce noise, etc.)
            #ruchika logic 
            #1 convert the audio file into .wav format into an audiosegment object
            #2converts audio into the mono (1channel) common for reducing file size or standardizing audio for analysis
            #3 changes the sample rate to 16Khz, which is often used speech processing models.
            #4 exporting the audio into .wav files after above modified
            
            audio_segment = AudioSegment.from_wav(audio_file)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)  # Convert to mono and set sample rate
            audio_segment.export("processed_audio.wav", format="wav")

            # Load processed audio and apply noise reduction
            #ruchika logic
            # rate is the sample rathe (in Hz) of the audio (16000Hz) and 
            # data is the actual audio signal data as Numpy array which hold the amplitude valuse of the audio Wave
            # nr.reduce_noise this method for the reduce audio
            #y=data specific the audio data to process
            #sr=data= tell the function the sample rate of the audiowhich helps to better model and remove the noise
            #then cleaned the noise againg update this audio in the files
            
            rate, data = wavfile.read("processed_audio.wav")
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            wavfile.write(f"processed_audio_clean{item['idx']}.wav", rate, reduced_noise)
            
            #return the update audio file
            os.remove(audio_file)  # Cleanup
            data={
                'audio_file':f"processed_audio_clean{item['idx']}.wav",
                'file_name':item['file_name'],
                'idx':item['idx'],
                'analysis_sheet':item['analysis_sheet'],
                 "video_url":item['video_url']
            }
            
            q_out.put(data)
            time.sleep(1)
            audio_file:f"processed_audio_clean{item['idx']}.wav"
        except Exception as e:
            pass
            #print(f"Error extracting audio: {e}")
           
# Function for speech-to-text transcription using Whisper
def transcribe_audio_whisper(q_in,q_out):
    while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
    
        try:
            audio_file=item['audio_file']
            # Whisper is video trancibe library, conver audio into text 
            #ruchika logic 
            #load basic model into the whisper
            #using trancribe it convert the audio into the text and return text if error ocurs then return the blank 
            
            model = whisper.load_model("base")  # You can also use "small", "medium", or "large" for better accuracy.
            result = model.transcribe(audio_file)
            
            data={
                 'trancribe_data':result["text"],
                'audio_file':audio_file,
                'file_name':item['file_name'],
                'idx':item['idx'],
                 "video_url":item['video_url'],
                'analysis_sheet':item['analysis_sheet']
            }
            print(data)
            q_out.put(data)
            time.sleep(1)
        except Exception as e:
            #print(f"Error transcribing audio: {e}")
            data={
                'trancribe_data':"",
                'audio_file':audio_file,
                'file_name':item['file_name'],
                 "video_url":item['video_url'],
                'idx':item['idx'],
                'analysis_sheet':item['analysis_sheet']
            }
            q_out.put(data)
            time.sleep(1)
def is_english_word(word):
    return word.lower() in cmu_dict
# Function to calculate the percentage of English words in the transcription
#function for the count the word in the text
def count_words(text):
    # retur the total words and total unique words
    if text:
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        unique_words = len(set(words))
        return total_words, unique_words
    else:
        return 0, 0
def calculate_english_percentage(q_in,q_out):
    while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
    
        try:
            transncription=item['trancribe_data']
            
            #ruchika logic here is 
            #using re library it cut text into the word and push into the list
            #for checking each world is english or not using nltk library 
            #calculate the percentage using of english world into the text 
            words = re.findall(r'\b\w+\b', transncription.lower())
            if not words:
                return 0  # Avoid division by zero if there are no words
            english_words = [word for word in words if is_english_word(word)]
            english_percentage = (len(english_words) / len(words)) * 100
            if english_percentage < 90:
                os.remove(item['audio_file'])
                os.remove(item['file_name'])
            else:
                total_duration = VideoFileClip(item['file_name']).duration
                total_words, unique_words = count_words(transncription)
                data={
                'total_duration':total_duration,
                'total_words':total_words,
                'unique_words':unique_words,
                'trancribe_data':transncription,
                'audio_file':item['audio_file'],
                 "video_url":item['video_url'],
                'file_name':item['file_name'],
                'idx':item['idx'],
                'analysis_sheet':item['analysis_sheet']
                }
                q_out.put(data)
                time.sleep(1)
                
        except Exception as e:
            os.remove(item['audio_file'])
            os.remove(item['file_name'])
            

# Function to detect pauses in audio
def detect_pauses(q_in,q_out):
     while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
        try:
            detected_language = langid.classify(item['trancribe_data'])[0]
            min_pause_duration=1000
            silence_thresh=-40
            audio_file=item['audio_file']
            audio = AudioSegment.from_wav(audio_file)
            pauses = silence.detect_silence(audio, min_silence_len=min_pause_duration, silence_thresh=silence_thresh)
            num_pauses = len(pauses)
            data={
                'detected_language':detected_language,
                'num_pauses':num_pauses,
                'total_duration':item['total_duration'],
                'total_words':item['total_words'],
                'unique_words':item['unique_words'],
                'trancribe_data':item['trancribe_data'],
                'audio_file':item['audio_file'],
                 "video_url":item['video_url'],
                'file_name':item['file_name'],
                'idx':item['idx'],
                'analysis_sheet':item['analysis_sheet']
            }
            q_out.put(data)
            time.sleep(1)
        except Exception as e:
            #print(f"Error detecting pauses: {e}")
            os.remove(item['audio_file'])
            os.remove(item['file_name'])
# Function to get average pitch using praat-parselmouth, focusing only on voiced segments
def get_average_pitch(q_in,q_out):
    while True:
        item=q_in.get()
        if item is None:
            q_out.put(None)
            break
        try:
            audio_file=item['audio_file']
            snd = parselmouth.Sound(audio_file)
            pitch = snd.to_pitch()
            # Calculate the average pitch without filtering, after background noise reduction
            pitch_values = pitch.selected_array['frequency']
            voiced_pitch_values = pitch_values[pitch_values > 0]  # Consider only voiced segments

            avg_pitch = np.mean(voiced_pitch_values) if len(voiced_pitch_values) > 0 else 0
            data={
                "avg_pitch":avg_pitch,
                'detected_language':item['detected_language'],
                'num_pauses':item['num_pauses'],
                'total_duration':item['total_duration'],
                'total_words':item['total_words'],
                'unique_words':item['unique_words'],
                'trancribe_data':item['trancribe_data'],
                'audio_file':item['audio_file'],
                 "video_url":item['video_url'],
                'file_name':item['file_name'],
                'idx':item['idx'],
                'analysis_sheet':item['analysis_sheet']
            }
            q_out.put(data)
            time.sleep(1)
        except Exception as e:
            pass
            #print(f"Error in pitch measurement: {e}")
        
        
# Function to analyze video frames for face data and brightness
def analyze_frames(q_in):
    while True:
        item=q_in.get()
        if item is None:
            
            break
        try:
            #convert the video file intot the open cv object to we can manipulate it
            #set the varibale
            video_file=item['file_name']
            duration=item['total_duration']
            
            cap = cv2.VideoCapture(video_file)
            face_detected_frames = 0
            brightness_values = []
            #find out the total frame from the cv2 library using => CAP_PROP_FRAME_COUNT
            #get frame rate then convert it tinto the integer
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) 
            sample_interval = fps  # Sample every 1 second based on FPS

            sampled_frames = range(0, total_frames, sample_interval)

            for i in sampled_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                #find out the basic metadata of the video which help to analysis the video
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray_frame)
                brightness_values.append(avg_brightness)
            
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    face_detected_frames += 1

            cap.release()
            #findout the matrics
            avg_brightness_overall = np.mean(brightness_values) if len(brightness_values) > 0 else 0
            face_detected_percentage = (face_detected_frames / len(sampled_frames)) * 100 if len(sampled_frames) > 0 else 0
            if face_detected_percentage <= 90:
                os.remove(item['audio_file'])
                os.remove(item['file_name'])
            else:
                result_row = [
                    item['video_url'], item['total_duration'], item['total_words'], item['unique_words'], item['trancribe_data'],
                    item['detected_language'], item['num_pauses'], item["avg_pitch"], len(sampled_frames), face_detected_frames,
                    face_detected_percentage, avg_brightness
                ]

                item['analysis_sheet'].append_row(result_row)
                os.remove(item['audio_file'])
                os.remove(item['file_name'])

            
        except Exception as e:
            pass
            

# def process_video(video_url, idx, analysis_sheet):
#     video_file = f"video_{idx+1}.mp4"
#     if 9==9:
#         function_metadata(download_video,video_url, video_file) #download_video(video_url,video_file)
        

#         # Step 1: Extract audio and transcription
#         audio_file =  function_metadata(extract_and_preprocess_audio,video_file)   #extract_and_preprocess_audio(video_file)
#         actual_transcription = function_metadata(transcribe_audio_whisper,audio_file)       #transcribe_audio_whisper(audio_file) # this function return the text 
        

#     #     # Step 2: Check if the transcription is mostly in English
#         english_percentage =function_metadata(calculate_english_percentage,actual_transcription)   #calculate_english_percentage(actual_transcription)
#         #this condition checks if the english word percentage in the video is less than 90% then remove it from system
#         if english_percentage < 90:
#             #print(f"Skipping video {video_url} as less than 90% of words are in English.")
#             os.remove(video_file)
#             os.remove(audio_file)
#             return

#         # Step 3: Analyze video for other metrics
#         total_duration = VideoFileClip(video_file).duration
#         #found the word into the text
        
#         total_words, unique_words = function_metadata(count_words,actual_transcription) #count_words(actual_transcription)
#         #detech the language using the text 
#         detected_language = langid.classify(actual_transcription)[0]
#         num_pauses = function_metadata(detect_pauses,audio_file) #detect_pauses(audio_file)
#         avg_pitch =function_metadata(get_average_pitch,audio_file) #get_average_pitch(audio_file)

#         (total_sampled_frames, face_detected_frames, face_detected_percentage, avg_brightness) = function_metadata(analyze_frames, video_file,total_duration)# analyze_frames(video_file, total_duration)
#         #print(total_sampled_frames, face_detected_frames, face_detected_percentage, avg_brightness)
#          # Only process videos where face is detected in more than 90% of frames
#          # we can make this function and process it  and fiflter out the criteria then update on the sheet the refresh data 
#         if face_detected_percentage <= 90:
#             #print(f"Skipping video {video_url} as face is not detected in more than 90% of frames.")
#             os.remove(video_file)
#             os.remove(audio_file)
#             return

#         result_row = [
#             video_url, total_duration, total_words, unique_words, actual_transcription,
#             detected_language, num_pauses, avg_pitch, total_sampled_frames, face_detected_frames,
#             face_detected_percentage, avg_brightness
#         ]

#         analysis_sheet.append_row(result_row)
#         os.remove(audio_file)
#         os.remove(video_file)
        
#     # except Exception as e:
#     #     #print(f"Error processing video {video_url}: {e}")
def process_videos_from_sheet():
    #get the data from sheet
    #url get from the Project analysis Tab 
    #for updating the result we use "Numerical Analysis" tab here
    sheet = client.open("Video Analysis")
    worksheet = sheet.worksheet("Project Assignment")
    data2D = worksheet.get_all_values()
    df=pd.DataFrame(data=data2D[1:], columns=data2D[0])
    
    

    try:
        analysis_sheet = sheet.worksheet("Numerical Analysis")
    except gspread.exceptions.WorksheetNotFound:
        analysis_sheet = sheet.add_worksheet(title="Numerical Analysis", rows="1000", cols="25")
        analysis_sheet.clear()

    headers = [
        "Video Link", "Total Duration (seconds)", "#Words", "#Unique Words", "Transcription",
        "Detected Language", "#Pauses", "Average Pitch (Hz)", "Total frames", "Face Detected Frames",
        "Face Detected Percentage", "Average Brightness"
    ]
    analysis_sheet.append_row(headers)
    #itrete for each video url make process for each url
    #in iterate call the process video function for the video analysis
    for i in range(len(df)): 
        data={
            'video_url':df.loc[i,'submission'],
             'idx':i,
             'analysis_sheet':analysis_sheet
        }
       
        queue1.put(data)
        #process_video(video_url, idx, analysis_sheet)
        time.sleep(1)
         
   
   
# this function is help to found out the what metadata it need 
def function_metadata(func, *args, **kwargs):
     # Record the initial CPU and memory usage
    process = psutil.Process()
    initial_cpu = process.cpu_percent(interval=None)
    tracemalloc.start()

    # Start time
    start_time = time.time()

    # Execute the function
    result = func(*args, **kwargs)

    # End time
    end_time = time.time()
    execution_time = end_time - start_time

    # Memory and CPU usage after function execution
    current, peak = tracemalloc.get_traced_memory()
    final_cpu = process.cpu_percent(interval=None)
    cpu_usage = final_cpu - initial_cpu
    memory_usage_kb = current / 1024  # Convert bytes to KB
    peak_memory_usage_kb = peak / 1024  # Convert bytes to KB

    # Stop tracing memory
    tracemalloc.stop()

    data={
    "Execution Time": f"{execution_time} seconds",
    "CPU Usage": f"{cpu_usage}%",
    "Memory Usage": f"{memory_usage_kb} KB",
    "Peak Memory Usage": f"{peak_memory_usage_kb}KB"
    }
    #print(data,func)
    return result

#define threads fo each function in the pipeline
threads = [
    threading.Thread(target= download_video, args=(queue1, queue2)),
    threading.Thread(target=extract_and_preprocess_audio, args=(queue2, queue3)),
    threading.Thread(target=transcribe_audio_whisper, args=(queue3, queue4)),
    threading.Thread(target=calculate_english_percentage, args=(queue4, queue5)),
    threading.Thread(target=detect_pauses, args=(queue5, queue6)),
    threading.Thread(target=get_average_pitch, args=(queue6,queue7)),
    threading.Thread(target=analyze_frames, args=(queue7,))
]
# Start all threads
for thread in threads:
    thread.start()

# Authenticate
try:
    client = authenticate_gspread('credentials.json')
   
except FileNotFoundError:
    #print("Error: Credentials file not found. Please check if the file was uploaded correctly.")
    exit()
    
    
# Process the videos from the Google Sheet
try:
    process_videos_from_sheet()
    queue1.put(None) # the signal for the first function stop
except KeyboardInterrupt:
    
    print("Process interrupted by user, resuming next video.")
    
# Wait for all threads to finish
for thread in threads:
    thread.join()

#print("All functions have completed.")