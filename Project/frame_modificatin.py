from imgaug import augmenters as iaa
import os
import cv2
import numpy as np

output_folder = r'C:\Users\stct\Documents\Modified Frames'

def load_batch(batch_idx, batch_size=102, image_size=(1920,1080), image_folder=r"C:\Users\stct\Documents\Original Frames"):
    """
    Loads a batch of images from the specified folder and returns them as a numpy array.
    :param batch_idx: Index of the batch to load.
    :param batch_size: Number of images to load in each batch.
    :param image_size: The target size to resize images (height, width).
    :param image_folder: Folder path where images are stored.
    :return: A numpy array of shape (batch_size, height, width, channels).
    """
    # Get a list of all image files in the folder (only .jpg, .png, .jpeg)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

    # Calculate the start and end indices of the batch
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size

    # Ensure we don't go out of bounds (if the batch_idx is too large)
    batch_files = image_files[start_idx:end_idx]
    
    images = []

    # Load each image in the batch, resize to the target size, and convert to uint8
    for image_file in batch_files:
        # Construct the full path to the image
        image_path = os.path.join(image_folder, image_file)

        # Read the image
        img = cv2.imread(image_path)

        if img is not None:
            # Resize image to the target size (if needed)
            img_resized = cv2.resize(img, image_size)

            # Ensure the image is in uint8 format (values between 0 and 255)
            images.append(img_resized)

    # Convert the list of images into a numpy array with shape (batch_size, height, width, channels)
    images = np.array(images, dtype=np.uint8)

    return images

def save_augmented_images(images_aug, output_folder, batch_idx, start_idx):
    """
    Save the augmented images to the output folder with a specific naming convention.
    """
    num_files = len(os.listdir(output_folder)) 

    for i, img in enumerate(images_aug):
        # Define the image file name
    
        output_file_name = f"augmented_batch_{batch_idx}_image_{start_idx + i+ num_files }.png"
        output_path = os.path.join(output_folder, output_file_name)

        # Save the augmented image
        cv2.imwrite(output_path, img)

augmentation_sequences = [
    iaa.Sequential([iaa.Rain(drop_size=(0.10, 0.20)), iaa.Fog()]),
    iaa.Sequential([iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))] * 3 + [iaa.Fog()]),
    iaa.Sequential([iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))] * 5),
    iaa.Sequential([iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)), iaa.GaussianBlur(sigma=(0.0, 3.0))]),
    iaa.Sequential([iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0, iaa.Add((0, 50))))]),
    iaa.Sequential([iaa.GammaContrast((0.5, 2.0), per_channel=True)]),
    iaa.Sequential([iaa.AllChannelsCLAHE()]),
    iaa.Sequential([iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)]),
    iaa.Sequential([iaa.imgcorruptlike.GaussianNoise(severity=1)]),
    iaa.Sequential([iaa.imgcorruptlike.GaussianNoise(severity=2)]),
    iaa.Sequential([iaa.imgcorruptlike.GaussianNoise(severity=3)]),
    iaa.Sequential([iaa.imgcorruptlike.ImpulseNoise(severity=2)]),
    iaa.Sequential([iaa.imgcorruptlike.ImpulseNoise(severity=3)]),
    iaa.FastSnowyLandscape(
        lightness_threshold=(100, 255),
        lightness_multiplier=(1.0, 4.0)),
    iaa.Sequential([iaa.imgcorruptlike.ImpulseNoise(severity=1)]),
    iaa.BlendAlphaSimplexNoise(
    foreground=iaa.BlendAlphaSimplexNoise(
        foreground=iaa.EdgeDetect(1.0),
        background=iaa.LinearContrast((0.5, 2.0)),
        per_channel=True
    ),
    background=iaa.BlendAlphaFrequencyNoise(
        exponent=(-2.5, -1.0),
        foreground=iaa.Affine(
            rotate=(-10, 10),
            translate_px={"x": (-4, 4), "y": (-4, 4)}
        ),
        background=iaa.AddToHueAndSaturation((-40, 40)),
        per_channel=True
    ),
    per_channel=True,
    aggregation_method="max",
    sigmoid=False
)
]

# Process batches and apply augmentations
for batch_idx in range(200):
    # Load the images for the current batch
    images = load_batch(batch_idx)

    # Apply each augmentation sequence in parallel
    for seq in augmentation_sequences:
        images_aug = seq(images=images)
        save_augmented_images(images_aug, output_folder, batch_idx, batch_idx * 32)



