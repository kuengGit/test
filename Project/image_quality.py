"""Module to manage frame quality indicator values to display in GUI."""
from enum import Enum
import math
import cv2
import numpy as np
import numpy.typing as npt

def pixels(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the number of pixels in a 3 channel image"""
    return frame.size/3

def brightness(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the mean pixel value
    Desired value - middle"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grey_frame)
    value = img_array.mean() / 255
    return value

def rms_contrast(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the aggregation of differences in each pixel to the mean value : the square root of all contents, where the contents is defined as the sum of all calculations divided by one less than the total number of pixels, where each calculation is defined as the square of a pixel's normalized value minus the normalized mean pixel value
    Desired value - middle"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grey_frame)
    img_array = img_array.astype(float)
    normalized_array = img_array / 255
    contrast = normalized_array.std()
    return contrast

def noise(frame: npt.NDArray[np.uint8]) -> float:
    """Measure the noise of the image using a filter to capture the high frequency variations
    Desired value - low"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grey_frame, dtype=np.double)
    height, width = img_array.shape
    laplat_filter = np.array([[1, -2, 1],
                              [-2, 4, -2],
                              [1, -2, 1]], dtype=np.double)

    convolution = cv2.filter2D(img_array, -1, cv2.flip(laplat_filter, -1), borderType = cv2.BORDER_CONSTANT)
    noise = np.sum(np.abs(convolution))    
    noise = noise * math.sqrt(0.5 * math.pi) / (6 * (width-2) * (height-2))
    return noise

def signal_to_noise(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the signal strength compared to noise for the image : the mean pixel value divided by the standard deviation of all pixels
    Desired value - high"""
    mean = frame.mean()
    std = frame.std()
    if std == 0.:
        return 0.
    return mean / std

def shannon_entropy(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the uncertainty in predicting pixel values : the negative sum of all calculations, where each calculation is defined as the probability for a pixel multiplied by log2 of that same probability
    Desired value - low"""
    r = cv2.calcHist([frame],[0],None,[256],[0,256]).ravel()
    g = cv2.calcHist([frame],[1],None,[256],[0,256]).ravel()
    b = cv2.calcHist([frame],[2],None,[256],[0,256]).ravel()
    histogram = r+g+b
    probabilities = histogram / np.prod(frame.shape)
    non_zero_probs = probabilities[probabilities != 0]
    entropy = - np.sum(non_zero_probs * np.log2(non_zero_probs))
    entropy = entropy.astype(float)
    return entropy

def canny_edge(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the number of edges detected indicating image sharpness
    Desired value - high"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.blur(grey_frame,ksize=(3,3))
    img_array = np.asarray(blurred_frame).astype(np.uint8)
    median_value = np.median(img_array)
    lower = int(max(0, 0.7 * median_value))
    upper = int(min(255, 1.3 * median_value))
    canny = cv2.Canny(image=img_array, threshold1=lower, threshold2=upper)
    edges = cv2.countNonZero(canny)
    return edges

def black_clipping(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the total quantity of pixels with a value of 0
    Desired value - low"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grey_frame)
    value = np.sum(img_array == 0) / np.prod(frame.shape)
    return value

def white_clipping(frame: npt.NDArray[np.uint8]) -> float:
    """Measure of the total quantity of pixels with a value of 255
    Desired value - low"""
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_array = np.asarray(grey_frame)
    value = np.sum(img_array == 255) / np.prod(frame.shape)
    return value

def saturation(frame):
    """Measure the average saturation of an image."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_frame[:, :, 1]
    value = saturation_channel.mean() / 255  # Normalize to 0-1
    return value


class ImageQualityMetrics(Enum):
    """
    Enum to represent image quality diagnostic functions
    """
    PIXELS = ("Pixels", pixels)
    BRIGHTNESS = ("Brightness", brightness)
    RMS_CONTRAST = ("RMS Contrast", rms_contrast)
    NOISE = ("Noise", noise)
    SIGNAL_TO_NOISE = ("Signal to Noise", signal_to_noise)
    SHANNON_ENTROY = ("Shannon Entropy", shannon_entropy)
    CANNY_EDGE = ("Canny Edge", canny_edge)
    BLACK_CLIPPING = ("Black Clipping", black_clipping)
    WHITE_CLIPPING = ("White Clipping", white_clipping)
    SATURATION = ("Saturation", saturation)
    
    @property
    def str_name(self):
        return self.value[0]

    @property
    def function(self):
        return self.value[1]
    

def evaluate_image_quality_metrics(frame: npt.NDArray[np.uint8]) -> dict[str, float]:
    evaluated_image_quality_metrics: dict[str, float] = {}

    # Validate 3 channel image for indicators
    if frame is not None and len(frame.shape) == 3: # type: ignore
        for image_diagnostic in ImageQualityMetrics:
            name: str = image_diagnostic.str_name
            evaluation: float = image_diagnostic.function(frame)
            evaluated_image_quality_metrics[name] = evaluation
    return evaluated_image_quality_metrics