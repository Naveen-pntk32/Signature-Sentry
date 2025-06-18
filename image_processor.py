import cv2
import numpy as np
import logging
from PIL import Image
import os

logger = logging.getLogger(__name__)

class SignatureImageProcessor:
    """Image preprocessing utilities for signature images"""
    
    def __init__(self, target_size=(105, 105)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path):
        """
        Preprocess signature image for neural network input
        
        Args:
            image_path: Path to the signature image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply image enhancements
            enhanced = self._enhance_signature(gray)
            
            # Resize to target size
            resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Add channel dimension for CNN
            processed = np.expand_dims(normalized, axis=-1)
            
            logger.debug(f"Preprocessed image shape: {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def _enhance_signature(self, gray_image):
        """
        Apply image enhancement techniques to improve signature quality
        
        Args:
            gray_image: Grayscale signature image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
            
            # Apply adaptive threshold to create binary image
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Invert colors so signature is white on black background
            inverted = cv2.bitwise_not(binary)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error enhancing signature image: {str(e)}")
            return gray_image
    
    def validate_image(self, image_path):
        """
        Validate if the uploaded file is a valid image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False, "Image file does not exist"
            
            # Check file size (max 5MB)
            file_size = os.path.getsize(image_path)
            if file_size > 5 * 1024 * 1024:
                return False, "Image file size too large (max 5MB)"
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                # Check if it's a valid image format
                img.verify()
            
            # Try to read with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return False, "Invalid image format"
            
            # Check minimum dimensions
            height, width = image.shape[:2]
            if height < 50 or width < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            
            return True, "Valid image"
            
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {str(e)}")
            return False, f"Image validation error: {str(e)}"
    
    def extract_signature_roi(self, image_path):
        """
        Extract Region of Interest (signature area) from the image
        
        Args:
            image_path: Path to the signature image
            
        Returns:
            numpy.ndarray: Cropped signature region
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not read image")
            
            # Apply threshold to create binary image
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assumed to be the signature)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop the signature region
                roi = image[y:y+h, x:x+w]
                return roi
            else:
                # If no contours found, return the original image
                return image
                
        except Exception as e:
            logger.error(f"Error extracting ROI from {image_path}: {str(e)}")
            # Return original image if ROI extraction fails
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def save_processed_image(self, processed_image, output_path):
        """
        Save processed image for debugging or display purposes
        
        Args:
            processed_image: Processed image array
            output_path: Path to save the image
        """
        try:
            # Convert back to 0-255 range if normalized
            if processed_image.max() <= 1.0:
                display_image = (processed_image * 255).astype(np.uint8)
            else:
                display_image = processed_image.astype(np.uint8)
            
            # Remove channel dimension if present
            if len(display_image.shape) == 3 and display_image.shape[-1] == 1:
                display_image = display_image.squeeze(-1)
            
            cv2.imwrite(output_path, display_image)
            logger.debug(f"Saved processed image to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")

# Global processor instance
image_processor = SignatureImageProcessor()
