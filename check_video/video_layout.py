from moviepy.editor import VideoFileClip, ImageClip, clips_array, CompositeVideoClip

class Video_modification:
    def  cut_video(video_clip,s_time,e_time):
       
        cut_clip=video_clip.subclip(s_time,e_time)
        target_aspect_ratio = 3/4
        orignal_w,orignal_h = cut_clip.size
        print(cut_clip.size)
        new_height = int(orignal_w / target_aspect_ratio)
        if new_height <orignal_h:
            final_clip = cut_clip.crop(y_center = orignal_h//2, height=new_height)
        else:
            final_clip = cut_clip.resize(height=int(orignal_w*target_aspect_ratio))   
        final_clip=final_clip.resize(newsize=(240, 380))    
        return final_clip
        
        print(final_clip.size)
    def get_video_duration(video_clip):
        
        # Get the duration of the video
        duration = video_clip.duration
        print(f"Video Duration: {duration} seconds")

        return duration
    def combine_video_image_video_right_insideimage(video_clip, image_clip):
       
        #load the img clip and set its duration to match the video
        image_clip = image_clip.set_duration(video_clip.duration) 
        # resize the imageto match the video width
        #create the final video v=by stacking them vertically video on top image below
        image_clip_resized = image_clip.resize(width=video_clip.w * 3.2, height=video_clip.h * 1.2)
        image_clip_resized = image_clip_resized.set_position('center')

        # Create a composite video with the video on top of the image
        final_clip = CompositeVideoClip([image_clip_resized, video_clip.set_position(('right', 'center'))])
        return final_clip
    def combine_video_image_image_inside_downvideo(video_clip,image_clip,padding=20):

        #image_clip = ImageClip(image_path)
        image_width = video_clip.w - 3 * padding
        image_clip_resized = image_clip.resize(width=image_width)
        image_clip_resized = image_clip.resize(width=video_clip.w * 0.5)
        image_clip_resized = image_clip_resized.set_duration(video_clip.duration)
        # Position the image at the bottom of the video, leaving padding from bottom
        image_y_position = video_clip.h - image_clip_resized.h - padding
        
        # Set the image's position with padding on all sides
        image_clip_resized = image_clip_resized.set_position(('center', image_y_position))

        # Composite the video and image, with video on top
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        print(final_clip)
        return final_clip

    def overlay_image_bottom_left(video_clip, image_clip, padding=20):
        image_clip = image_clip.set_duration(video_clip.duration)
        image_clip_resized = image_clip.resize(width=video_clip.w * 0.2)
        # Position the image in the bottom-left corner with padding
        image_clip_resized = image_clip_resized.set_position((padding, video_clip.h - image_clip_resized.h - padding))
        # Composite the video and image
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        return final_clip
    def overlay_transparent_image_bottom(video_clip, image_clip,  padding=20, opacity=0.6):
        image_clip = image_clip.set_duration(video_clip.duration).resize(width=video_clip.w * 0.4)
        
        # Add transparency (opacity)
        image_clip_resized = image_clip.set_opacity(opacity)
        
        # Position the image in the bottom-center
        image_clip_resized = image_clip_resized.set_position(('center', video_clip.h - image_clip_resized.h - padding))

        # Composite the video and image
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        return final_clip
    def split_screen_image_right(video_clip, image_clip, output_path):
        
        # Resize video to take the left half of the screen
        video_resized = video_clip.resize(width=video_clip.w // 2)

        # Resize image to fit the right half
        image_clip = image_clip.set_duration(video_clip.duration)
        image_resized = image_clip.resize(width=video_resized.w, height=video_resized.h)
        
        # Set the image to the right half
        image_resized = image_resized.set_position((video_resized.w, 0))

        # Composite the video on the left and image on the right
        final_clip = CompositeVideoClip([video_resized.set_position((0, 0)), image_resized])
        return final_clip
    def fade_in_out_image(video_clip, image_clip, fade_duration=2):
        image_clip = image_clip.set_duration(video_clip.duration)

        # Resize image to be smaller
        image_clip_resized = image_clip.resize(width=video_clip.w * 0.4)
        
        # Add fading effect
        image_clip_resized = image_clip_resized.fadein(fade_duration).fadeout(fade_duration)

        # Position image in the center
        image_clip_resized = image_clip_resized.set_position(('center', 'center'))
        
        # Composite the video and image
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        return final_clip
    def slide_in_image_from_right(video_clip, image_clip, slide_duration=2):
        
        image_clip = image_clip.set_duration(video_clip.duration)

        # Resize image to be smaller
        image_clip_resized = image_clip.resize(width=video_clip.w * 0.3)
        
        # Animate the image sliding in from the right
        image_clip_resized = image_clip_resized.set_position(lambda t: ('center', video_clip.h * 1.2 - t * slide_duration * 100))

        # Composite the video and image
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        return final_clip
    def watermark_image_top_right(video_clip, image_clip, padding=20):
        
        image_clip = image_clip.set_duration(video_clip.duration)
        image_clip_resized = image_clip.resize(width=video_clip.w * 0.15)
        
        # Position the image in the top-right corner
        image_clip_resized = image_clip_resized.set_position((video_clip.w - image_clip_resized.w - padding, padding))
        
        # Composite the video and image
        final_clip = CompositeVideoClip([video_clip, image_clip_resized])
        return final_clip

        
