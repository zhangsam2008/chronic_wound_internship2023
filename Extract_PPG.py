import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from patchify import patchify,unpatchify

def segment_feet(img, K):
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

def average_img_patches(segemented_img, win_size, channel, step):
    win_size = win_size
    channel = channel
    step = step

    segment_patch = patchify(segemented_img, (win_size, win_size), step)
    segment_patch_mean = np.mean(segment_patch, (2, 3))
    segment_patch_std = np.std(segment_patch, (2, 3))
    print("patch dim:", segment_patch_mean.shape, segment_patch.shape, segment_patch_std.shape)

    segemented_patch_zeros = np.zeros(segment_patch.shape)

    # segment_patch_mean = np.repeat(segment_patch_mean[:,:,None,:],1,axis=2)
    segment_patch_mean = np.repeat(segment_patch_mean[:, :, None], win_size, axis=2)
    segment_patch_mean = np.repeat(segment_patch_mean[:, :, :, None], win_size, axis=3)

    # segment_patch_std = np.repeat(segment_patch_std[:,:,None,:],1,axis=2)
    segment_patch_std = np.repeat(segment_patch_std[:, :, None], win_size, axis=2)
    segment_patch_std = np.repeat(segment_patch_std[:, :, :, None], win_size, axis=3)

    print("patch dim:", segment_patch_mean.shape, segment_patch.shape, segment_patch_std.shape)

    segment_patch_ = segemented_patch_zeros + segment_patch_mean

    segment_patch = unpatchify(segment_patch_, segemented_img.shape)

    return segment_patch


def ppg_amplitude_mapping(img1, img2, mask, sequence):
    sequence = (sequence - np.mean(sequence)) / (np.std(sequence))
    # img_diff = img_diff/((np.sum(sequence**2)/sequence.shape[0])**(1/2))

    background_label = mask[0][-1]
    clusters = np.unique(mask)
    pulsation_map = []

    for c in clusters:
        if c == background_label:
            continue
        print("cluster: #%d" % c)

        cluster_im = np.zeros(img1.shape)
        obj_indices = np.where(mask == c)
        cluster_im[obj_indices] = (np.abs(img1[obj_indices] - img2[obj_indices]) ** 2) / (
                    (np.sum(sequence ** 2) / sequence.shape[0]) ** (1 / 2))

        start = np.min(cluster_im[obj_indices])
        end = np.max(cluster_im[obj_indices])

        cluster_im[obj_indices] = (cluster_im[obj_indices] - np.mean(cluster_im[obj_indices])) / (
            np.std(cluster_im[obj_indices]))

        padded_nid = np.reshape(cluster_im, (-1, 1))
        padded_nid = np.repeat(padded_nid, sequence.shape[0], axis=1)
        padd_seq = np.repeat(sequence[None, :], padded_nid.shape[0], axis=0)
        padded_nid = np.argmin(np.abs(padded_nid - padd_seq) * 2, axis=1)

        # print(padded_nid)

        padded_nid = sequence[padded_nid]
        padded_nid = padded_nid.reshape(cluster_im.shape)
        pulsation_map.append(padded_nid)
    pulsation_map = np.array(pulsation_map)
    pulsation_map = np.max(pulsation_map, axis=0)
    return pulsation_map

def process_img(img, win_size=5, channel=3, step=1, clean_area=-1, K=3, direction="x"):
    mask, label = segment_feet(img, K)
    background_label = label[0][-1]

    obj = np.where(label != background_label)

    temp_mask = np.zeros(label.shape)
    temp_mask[obj] = 1
    if direction == "x":
        temp_mask[:, 0:100] = 0
        temp_mask[:, clean_area:] = 0
    else:
        temp_mask[clean_area:, :] = 0

    temp_mask = np.repeat(temp_mask[:, :, None], 3, axis=2)
    segemented_img = img * temp_mask

    print(segemented_img[:, :, 1].shape)
    segemented_img = average_img_patches(segemented_img[:, :, 1], win_size, channel, step)
    return segemented_img, label, img


def extract_green_frames(video_frames_dir, output_dir, win_size, channel, step, clean_area, K):
    video_filenames = sorted(os.listdir(video_frames_dir))
    n = 1
    for video_filename in video_filenames:
        print(f"Processing frame {n}")
        video_frame_path = os.path.join(video_frames_dir, video_filename)
        video_frame = cv2.imread(video_frame_path, cv2.IMREAD_UNCHANGED)
        if video_frame is None:
            print(f"Failed to load video frame: {video_frame_path}")
            continue
        img1, label1, ori_img1 = process_img(video_frame, win_size, channel, step, clean_area, K)
        img1_g = img1.copy()
        output_filename = f"frames_{n}.png"
        output_path = os.path.join(output_dir, output_filename)
        # img1 = img1.astype(np.uint8)
        # img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        plt.imsave(output_path, img1_g)
        n += 1
    print("extract green done")

def compute_ppg_signal(frames_directory, roi_start, roi_end):
    video_green_channel = []
    for i in range(1, len(os.listdir(frames_directory)) + 1):
        frame_path = os.path.join(frames_directory, f"frames_{i}.png")
        frame = cv2.imread(frame_path)[:,:,1]  # Extract the green channel
        video_green_channel.append(frame)

    video_green_channel = np.array(video_green_channel)
    video_green_channel = video_green_channel[100:]

    traditional_ppg = video_green_channel[:, roi_start:roi_end, roi_start:roi_end]
    traditional_ppg_res = np.mean(traditional_ppg, axis=(1, 2))[90:125]
    traditional_ppg_res = (traditional_ppg_res - np.min(traditional_ppg_res)) / (
                np.max(traditional_ppg_res) - np.min(traditional_ppg_res))

    plt.plot(traditional_ppg_res)
    plt.xlabel('Frame')
    plt.ylabel('PPG Signal')
    plt.title('Computed PPG Signal')
    plt.show()
    return traditional_ppg_res

def find_frequency_range(rppg_signals, sampling_rate):
    # Perform Fourier Transform on the rPPG signals
    fft_signal = np.fft.fft(rppg_signals)

    # Compute the power spectrum
    power_spectrum = np.abs(fft_signal) ** 2

    # Determine the frequency resolution
    freq_resolution = sampling_rate / len(rppg_signals)

    # Find the main energy frequency range
    main_energy_start = int(0.5 / freq_resolution)  # Lower frequency limit (e.g., 0.5 Hz)
    main_energy_end = int(20 / freq_resolution)  # Upper frequency limit (e.g., 20 Hz)

    # Extract the main energy spectrum within the frequency range
    main_energy_spectrum = power_spectrum[main_energy_start:main_energy_end]

    # Find the index of the peak frequency
    peak_index = np.argmax(main_energy_spectrum)

    # Check for consecutive rising frequencies
    consecutive_rise = 0
    max_consecutive_rise = 0
    max_consecutive_rise_index = peak_index

    for i in range(peak_index, len(main_energy_spectrum) - 1):
        if max_consecutive_rise * freq_resolution >= 0.1:
            if main_energy_spectrum[i + 1] < main_energy_spectrum[i]:
                break
            else:
                continue
        if main_energy_spectrum[i + 1] > main_energy_spectrum[i]:
            consecutive_rise += 1
            if consecutive_rise > max_consecutive_rise:
                max_consecutive_rise = consecutive_rise
                max_consecutive_rise_index = i + 1
        else:
            consecutive_rise = 0

    # Calculate the peak frequency in Hz if consecutive rise is sufficient
    if max_consecutive_rise * freq_resolution >= 0.1:
        print("Consecutive rise detected")
        peak_frequency = (main_energy_start + max_consecutive_rise_index) * freq_resolution
    else:
        peak_frequency = (main_energy_start + peak_index) * freq_resolution

    # Plot the power spectrum
    frequencies = np.arange(len(power_spectrum)) * freq_resolution
    plt.plot(frequencies, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectrum')
    plt.axvline(x=peak_frequency, color='r', linestyle='--', label='Peak Hz')
    plt.legend()
    plt.show()

    print("Peak Frequency (Hz):", peak_frequency)
    return peak_frequency
def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    frame_counter = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Save the frame as an image
        frame_filename = f"frames_{frame_counter}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        frame_counter += 1

    # Release the video file
    video.release()

    print("Frames extraction completed.")

def Find_PPG(video_path, fps=20,output_dir='video_frames', win_size=5, channel=3, step=1, clean_area=-1, K=3, roi_start=1000, roi_end=1009):
    extract_frames(video_path, output_dir)
    green_dir = os.path.join("green")
    if not os.path.exists(green_dir):
        os.makedirs(green_dir)
    extract_green_frames(output_dir, green_dir, win_size, channel, step, clean_area, K)
    ppg = compute_ppg_signal('green', roi_start, roi_end)
    peak_frequency=find_frequency_range(ppg, fps)
    return peak_frequency

if __name__ == '__main__':
    Find_PPG('right_940_20fps_gain4_polarized_synced.avi')


