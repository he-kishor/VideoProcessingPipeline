import subprocess

def extract_audio(input_video_path, output_audio_path):
    # FFmpeg command to extract audio from video
    command = [
        'ffmpeg', '-i', input_video_path, 
        '-q:a', '0', '-map', 'a', output_audio_path
    ]
    
    # Execute the command
    subprocess.run(command, check=True)

def add_subtitles(input_video_path, subtitles_path, output_video_path):
    # FFmpeg command to add subtitles to the video
    command = [
        'ffmpeg', '-i', input_video_path,    # Input video file
        '-i', subtitles_path,                 # Subtitles file
        '-c', 'copy',                         # Copy the video and audio streams
        '-c:s', 'mov_text',                   # Set the codec for the subtitles
        output_video_path                     # Output video file
    ]
    
    # Execute the command
    subprocess.run(command, check=True)


# Example usage
input_video = 'testing_f.mp4'  # Replace with your input video file
output_audio = 'output.mp3'  # Replace with your desired output audio file


# Example usage      # Replace with your input video file
#
# subtitles_file = 'subtitles.srt'   # Replace with your subtitles file
output_video = 'output.mp4'        # Replace with your desired output video file

# try:
#     add_subtitles(input_video, subtitles_file, output_video)
#     print(f"Subtitles added successfully. Saved to {output_video}")
# except subprocess.CalledProcessError as e:
#     print("Error during subtitle addition:", e)

# try:
#     extract_audio(input_video, output_audio)
#     print(f"Audio extracted successfully and saved to {output_audio}")
# except subprocess.CalledProcessError as e:
#     print("Error during audio extraction:", e)
