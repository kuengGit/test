import cv2
import numpy as np
import random

def add_raindrops(image, num_drops=50, max_drop_size=50, min_drop_size=10):
    """
    Adds a raindrop effect to the image to simulate raindrops on a lens.
    :param image: Input image (numpy array).
    :param num_drops: Number of raindrops to add.
    :param max_drop_size: Maximum size of the raindrops.
    :param min_drop_size: Minimum size of the raindrops.
    :return: Image with raindrop effect.
    """
    height, width, _ = image.shape
    
    # Convert the image to a float32 array to handle transparency and manipulation
    image_with_drops = image.copy().astype(np.float32)
    
    for _ in range(num_drops):
        # Randomly choose a position and size for the raindrop
        drop_radius = random.randint(min_drop_size, max_drop_size)
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        
        # Create a circular mask for the raindrop
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= drop_radius
        
        # Create the raindrop: a semi-transparent blurry circle
        drop_intensity = random.uniform(0.1, 0.3)  # Adjust transparency of the raindrop
        blur_radius = random.randint(3, 10)  # Blurriness of the raindrop
        drop_color = np.array([0, 0, 255])  # Red color (can be adjusted)

        # Apply Gaussian blur to simulate the blurry edges of the drop
        raindrop = np.zeros_like(image, dtype=np.float32)
        raindrop[mask] = drop_color
        
        # Ensure the blur_radius is odd
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        
        # Blur the raindrop for more realism
        raindrop_blurred = cv2.GaussianBlur(raindrop, (blur_radius, blur_radius), 0)

        # Apply the raindrop effect to the image
        image_with_drops = cv2.addWeighted(image_with_drops, 1, raindrop_blurred, drop_intensity, 0)
    
    # Convert back to uint8 (original image type)
    image_with_drops = np.clip(image_with_drops, 0, 255).astype(np.uint8)

    # Apply a slight overall blur to simulate focus loss due to lens distortion
    image_with_drops = cv2.GaussianBlur(image_with_drops, (7, 7), 0)
    
    return image_with_drops

# Load an example image
image_path = r'C:\Users\stct\Documents\Original Frames\frame_1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if image was loaded properly
if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Add raindrop effect
    image_with_raindrops = add_raindrops(image, num_drops=100)

    # Show the image
    cv2.imshow("Raindrops Effect", image_with_raindrops)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
