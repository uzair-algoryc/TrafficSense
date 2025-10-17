from moviepy.editor import VideoFileClip

clip = VideoFileClip("/home/algoryc/Downloads/License_Plate.mov")
print(f"Duration: {clip.duration} seconds")

clip.subclip(0, 25).write_videofile("license_plate.mp4")

