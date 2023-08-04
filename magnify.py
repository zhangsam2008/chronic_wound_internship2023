from Extract_PPG import Find_PPG
from Extract_Vessel import Vessel_Extract_Video, Overlay_Vessel_Video,segment_feet,frames_to_video,segment_video
from PyEVM import magnify_color, magnify_motion

def magnify(video):
    fre=Find_PPG(video)
    low=(fre-0.2)/2
    high=(fre+0.2)/2
    Vessel_Extract_Video('right_940_20fps_gain4_polarized_synced.avi','vessel.mp4')
    frames_to_video('vessel_frames','vessel.mp4')
    frames_to_video('video_frames','video.mp4')
    magnify_color('vessel.mp4','magnifiedvessl_frames.mp4',low,high)

    Overlay_Vessel_Video('magnifiedvessl_frames.mp4', 'video.mp4', 'overlap.mp4')
    #magnify_motion('overlap.mp4','magnifiedoverlap.mp4',1.5/2.5,1.7/2.5)
if __name__ == '__main__':
    magnify('0524 patient 0304.avi')