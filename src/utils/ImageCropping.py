import cv2
import numpy as np
def crop_image(image_input):
    num_envs = image_input.shape[0]
    num_frames = image_input.shape[1]
    output_data = np.zeros((num_envs, num_frames,84, 84), dtype=np.float64)
    # Loop through each sample in the input data
    for ens_idx in range(num_envs):
        for frame_idx in range(num_frames):
            # Resize the input image to (84, 84)
            resized_image = cv2.resize(image_input[ens_idx, frame_idx], (84, 84))

            # Convert the resized image to grayscale
            grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


            # divide by 255.0
            grayscale_image = grayscale_image / 255.0

            # Assign the processed image to the output array
            output_data[ens_idx, frame_idx] = grayscale_image

    return output_data


