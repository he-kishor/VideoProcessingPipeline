from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
from moviepy.config import change_settings

# Set the ImageMagick path again (optional if the previous step worked)
change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.0.10-Q16-HDRI\\magick.exe"})

# Load your base video
videoclip = VideoFileClip('testing23.mp4')

# Add text to the video
text = TextClip("Your Text Here", fontsize=50, color='white', font='Amiri-Bold')
text = text.set_position(("center", "bottom")).set_duration(videoclip.duration)

# Combine video and text
final_video = CompositeVideoClip([videoclip, text])

# Export the final video
final_video.write_videofile("output_video.mp4", codec="libx264")