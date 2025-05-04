import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:  # Already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(image: np.ndarray, method: str = 'gaussian', kernel_size: int = 5) -> np.ndarray:
    try:
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif method == 'nlm':  # Non-local means denoising
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            logger.warning(f"Unknown denoising method: {method}, using gaussian")
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except Exception as e:
        logger.error(f"Error in denoising: {e}")
        return image  # Return original if denoising fails

def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    try:
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        elif method == 'hist_eq':
            return cv2.equalizeHist(image)
        else:
            logger.warning(f"Unknown contrast enhancement method: {method}, using CLAHE")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    except Exception as e:
        logger.error(f"Error in contrast enhancement: {e}")
        return image  # Return original if enhancement fails

def threshold_image(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    try:
        if method == 'otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == 'adaptive':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == 'binary':
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return thresh
        else:
            logger.warning(f"Unknown thresholding method: {method}, using Otsu")
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
    except Exception as e:
        logger.error(f"Error in thresholding: {e}")
        return image  # Return original if thresholding fails

def edge_detection(image: np.ndarray, method: str = 'canny', low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    try:
        if method == 'canny':
            return cv2.Canny(image, low_threshold, high_threshold)
        elif method == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(sobelx, sobely).astype(np.uint8)
        else:
            logger.warning(f"Unknown edge detection method: {method}, using Canny")
            return cv2.Canny(image, low_threshold, high_threshold)
    except Exception as e:
        logger.error(f"Error in edge detection: {e}")
        return np.zeros_like(image)  # Return empty image if edge detection fails

def preprocess_for_text(image: np.ndarray) -> np.ndarray:
    try:
        # Convert to grayscale
        gray = to_grayscale(image)

        # Denoise
        denoised = denoise(gray, method='gaussian', kernel_size=3)

        # Enhance contrast
        enhanced = enhance_contrast(denoised, method='clahe')

        # Apply thresholding to get binary image
        # Note: For OCR sometimes inverse binary (white text on black) works better
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return to_grayscale(image)  # Return simple grayscale as fallback

def preprocess_for_indicators(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a grayscale version for contour detection
        gray = to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold for contour detection
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return hsv, thresh
    except Exception as e:
        logger.error(f"Error in indicator preprocessing: {e}")
        # Return fallbacks
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
        return hsv, to_grayscale(image)

def preprocess_for_dials(image: np.ndarray) -> np.ndarray:
    try:
        # Convert to grayscale
        gray = to_grayscale(image)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection for circle detection
        edges = cv2.Canny(blurred, 50, 150)

        return edges
    except Exception as e:
        logger.error(f"Error in dial preprocessing: {e}")
        return to_grayscale(image)  # Return simple grayscale as fallback

def preprocess_all(image: np.ndarray) -> Dict[str, np.ndarray]:
    results = {}

    # Store original
    results['original'] = image.copy()

    # Basic preprocessing
    results['grayscale'] = to_grayscale(image)
    results['denoised'] = denoise(results['grayscale'])
    results['enhanced'] = enhance_contrast(results['denoised'])
    results['binary'] = threshold_image(results['enhanced'])
    results['edges'] = edge_detection(results['enhanced'])

    # Special-purpose preprocessing
    results['text_ready'] = preprocess_for_text(image)
    results['hsv'], results['indicator_ready'] = preprocess_for_indicators(image)
    results['dial_ready'] = preprocess_for_dials(image)

    return results

def main(image: np.ndarray) -> Dict[str, np.ndarray]:
    return preprocess_all(image)

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Preprocess an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        exit(1)

    # Apply preprocessing
    results = main(image)

    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (name, img) in enumerate(results.items()):
        if i >= len(axes):
            break

        if name == 'hsv':
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        elif len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')

        axes[i].set_title(name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
