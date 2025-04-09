import cv2
import numpy as np
import random

def add_frozen_effect(image, blur_strength=15, frost_intensity=0.4):
    """
    Adds a frozen lens effect (like condensation on the lens) to the image.
    :param image: Input image (numpy array).
    :param blur_strength: Strength of the Gaussian blur to simulate lens focus loss.
    :param frost_intensity: Intensity of the frosty texture effect.
    :return: Image with frozen lens effect.
    """
    height, width, _ = image.shape

    # Step 1: Apply Gaussian blur to simulate the blurry, foggy appearance of a frozen lens.
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # Step 2: Create a frosty texture (random noise) to simulate condensation on the lens.
    frost_texture = np.random.normal(loc=0, scale=255 * frost_intensity, size=(height, width, 3)).astype(np.uint8)
    
    # Step 3: Blend the frosty texture with the blurred image.
    # The intensity of the frost effect will depend on the "frost_intensity" parameter.
    frozen_image = cv2.addWeighted(blurred_image, 1 - frost_intensity, frost_texture, frost_intensity, 0)
    
    return frozen_image

# Load an example image
image_path = r'C:\Users\stct\Documents\Original Frames\frame_1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if image was loaded properly
if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Add frozen effect
    image_with_frozen_effect = []
    for fost_int in  np.arange(0, 0.5, 0.05):
        image_with_frozen_effect.append(add_frozen_effect(image, blur_strength=15, frost_intensity=fost_int))

    # Show the image with the frozen effect
    if __name__ == "__main__":
        cv2.imshow("Frozen Lens Effect", image_with_frozen_effect[4])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
