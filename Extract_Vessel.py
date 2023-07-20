import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from skimage.filters import meijering, sato, frangi, hessian
from matplotlib.colors import LinearSegmentedColormap
import os
from natsort import natsorted


def segment_feet(img, K=3):
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    label = label.reshape(img.shape[:2])
    result_image = res.reshape((img.shape))

    return result_image, label

def vascular_map(img_,black_ridges=True,sigmas=14,method=sato):
    result = method(img_, black_ridges=black_ridges, sigmas=sigmas)
    vascular_pixel = np.where(result!=0)
    vessels = result[vascular_pixel]
    #print(np.std(vessels))
    result[vascular_pixel] = (vessels - np.min(vessels))/(np.max(vessels)-np.min(vessels))
    return result

def removed_background(img_label,after_x):
    mask = np.zeros(img_label.shape)
    background_label = img_label[0][-1]
    obj = np.where(img_label!=background_label)
    mask[obj] = 1
    mask[:,after_x:]=0
    mask = np.repeat(mask[:,:,None],3,axis=-1)
    return mask

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def remove_border(im, left, right, bottom, top):
    for pixel in range(left):
        left = first_nonzero(im, 1)
        boundry = (np.arange(0, left.shape[0]), left)
        im[boundry] = 0

    for pixel in range(right):
        right = last_nonzero(im, 1)
        boundry = (np.arange(0, right.shape[0]), right)
        im[boundry] = 0

    for pixel in range(bottom):
        bottom = last_nonzero(im, 0)
        boundry = (bottom, np.arange(0, bottom.shape[0]))
        im[boundry] = 0

    for pixel in range(top):
        top = first_nonzero(im, 0)
        boundry = (top, np.arange(0, top.shape[0]))
        im[boundry] = 0

    return im

def Vessel_Extract_Video(inputvideo,outputpath):
    # Open the input video
    cap = cv2.VideoCapture(inputvideo)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object to save the output video
    # fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(outputpath, fourcc, fps, size)

    frame_counter = 1

    colors = [(0, 0, 0), (1, 0, 0)]
    cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        im = frame[:-1 * (frame.shape[0] % 100), :-1 * (frame.shape[1] % 100)]
        # Segment feet in the frame
        res_im, im_label = segment_feet(im)
        # Compute the mask for the feet
        # mask_im = removed_background(im_label, 1300)
        # im = im * mask_im
        # frame_resized = im
        # im = im[:, :1300]
        # Compute the vessel map
        im_green = np.int64(im[:, :, 1])
        im_idx = np.where(im_green > 0)
        tile_pixel = np.min(im_green[im_idx])
        background_idx = np.where(im_green == 0)
        im_green[background_idx] = tile_pixel
        vascular_frame = vascular_map(im_green, sigmas=range(10, 14), method=sato)
        # Convert the grayscale image to 3-channel
        vascular_filename = f"vessels_{frame_counter}.png"
        vascular_path = os.path.join('vessel_frames', vascular_filename)
        frames_filename = f"frames_{frame_counter}.png"
        frames_path = os.path.join('video_frames', frames_filename)
        plt.imsave(vascular_path,vascular_frame, cmap=cm)
        plt.imsave(frames_path,im_green, cmap="gray")
        # out.write(vascular_frame)
        print(f"Processing frame {frame_counter}/{total_frames}")
        frame_counter += 1
    # Release video capture and writer
    cap.release()
    # out.release()

    print("Video processing complete.")

from moviepy.editor import ImageSequenceClip

def Overlay_Vessel_Video(vessel_video_path, video_path, output_video):
    # Convert videos to frames
    vessel_video = VideoFileClip(vessel_video_path)
    video = VideoFileClip(video_path)

    vessel_frames = [frame for frame in vessel_video.iter_frames()]
    video_frames = [frame for frame in video.iter_frames()]

    frame_width, frame_height = video_frames[0].shape[1], video_frames[0].shape[0]
    fps = min(vessel_video.fps, video.fps)

    overlaid_frames = []
    for i, (vessel_frame, video_frame) in enumerate(zip(vessel_frames, video_frames)):
        print(f"Processing frame {i + 1}")

        vessel_frame = cv2.cvtColor(vessel_frame, cv2.COLOR_RGB2BGR)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)

        # Ensure the vessel frame and the video frame have the same shape
        if vessel_frame.shape != video_frame.shape:
            print(f"Frame {i+1} shape mismatch, resizing vessel frame to match video frame")
            vessel_frame = cv2.resize(vessel_frame, (video_frame.shape[1], video_frame.shape[0]))

        # Overlay the frames
        overlaid_frame = cv2.addWeighted(video_frame, 0.4, vessel_frame, 0.6, 1.5)
        overlaid_frame = cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)
        overlaid_frames.append(overlaid_frame)

    # Create video clip from overlaid frames
    if overlaid_frames:
        overlaid_video = ImageSequenceClip(overlaid_frames, fps=fps)
        overlaid_video.write_videofile(output_video, codec='libx264', fps=fps)
    frames_to_video('vessel_frames', 'vessel_video.mp4')
    print("Done")
def frames_to_video(frames_dir, output_path, output_fps=20):
    # Get the list of image frames in the directory
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

    # Read the first frame to get dimensions
    frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, channels = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    # Write each frame to the video file
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video saved to: {output_path}")

def segment_video(input_dir_path, output_dir_path):
    filenames = natsorted(os.listdir(input_dir_path))

    for filename in filenames:
        input_filepath = os.path.join(input_dir_path, filename)
        img = cv2.imread(input_filepath)
        if img is None:
            print(f"Failed to load image: {input_filepath}")
            continue

        result_image, label = segment_feet(img, K=3)
        output_filepath = os.path.join(output_dir_path, filename)
        cv2.imwrite(output_filepath, result_image)
def main():
    inputvideo = 'right_940_20fps_gain4_polarized_synced.avi'
    outputpath = 'vessel.mp4'
    Vessel_Extract_Video(inputvideo,outputpath)
if __name__ == '__main__':
    main()
    print("Done")