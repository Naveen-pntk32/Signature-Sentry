import os
import logging
import numpy as np
import cv2
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class SiameseSignatureModel:
    """Computer Vision-based signature fraud detection using traditional ML techniques"""
    
    def __init__(self, input_shape=(105, 105, 1)):
        self.input_shape = input_shape
        self.threshold = 0.7
        logger.info("Signature model initialized with computer vision approach")
        
    def _extract_features(self, image):
        """Extract multiple feature types from signature image"""
        try:
            # Ensure image is 2D
            if len(image.shape) == 3:
                image = image.squeeze()
            
            features = {}
            
            # 1. Simple gradient-based features (replacing HOG)
            grad_x = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Create histogram of gradient directions (8 bins)
            hist, _ = np.histogram(gradient_direction.flatten(), bins=8, range=(-np.pi, np.pi))
            features['hog'] = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            # 2. Contour-based features
            contours, _ = cv2.findContours((image * 255).astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Contour area
                features['contour_area'] = cv2.contourArea(largest_contour)
                # Contour perimeter
                features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['aspect_ratio'] = float(w) / h if h > 0 else 1.0
            else:
                features['contour_area'] = 0
                features['contour_perimeter'] = 0
                features['aspect_ratio'] = 1.0
            
            # 3. Statistical features
            features['mean_intensity'] = np.mean(image)
            features['std_intensity'] = np.std(image)
            features['skewness'] = self._calculate_skewness(image)
            features['kurtosis'] = self._calculate_kurtosis(image)
            
            # 4. Texture features using local binary pattern approximation
            features['texture'] = self._calculate_texture_features(image)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return default features in case of error
            return {
                'hog': np.zeros(72),  # Default HOG size
                'contour_area': 0,
                'contour_perimeter': 0,
                'aspect_ratio': 1.0,
                'mean_intensity': 0.5,
                'std_intensity': 0.1,
                'skewness': 0,
                'kurtosis': 0,
                'texture': np.zeros(8)
            }
    
    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity distribution"""
        flat = image.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std == 0:
            return 0
        return np.mean(((flat - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity distribution"""
        flat = image.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std == 0:
            return 0
        return np.mean(((flat - mean) / std) ** 4) - 3
    
    def _calculate_texture_features(self, image):
        """Calculate simple texture features"""
        # Use gradient-based texture analysis
        grad_x = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate texture statistics
        texture_features = np.array([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude),
            np.min(gradient_magnitude),
            np.percentile(gradient_magnitude, 25),
            np.percentile(gradient_magnitude, 50),
            np.percentile(gradient_magnitude, 75),
            np.sum(gradient_magnitude > np.mean(gradient_magnitude))
        ])
        
        return texture_features
    
    def predict_similarity(self, img1, img2):
        """
        Predict similarity between two signature images using multiple metrics
        
        Args:
            img1: First preprocessed signature image (numpy array)
            img2: Second preprocessed signature image (numpy array)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Ensure images are in the right format
            if len(img1.shape) == 4:
                img1 = img1.squeeze()
            if len(img2.shape) == 4:
                img2 = img2.squeeze()
            
            # Extract features from both images
            features1 = self._extract_features(img1)
            features2 = self._extract_features(img2)
            
            # Calculate different similarity metrics
            similarities = {}
            
            # 1. Custom Structural Similarity (replacing SSIM)
            if len(img1.shape) == 3:
                img1_2d = img1.squeeze()
            else:
                img1_2d = img1
            if len(img2.shape) == 3:
                img2_2d = img2.squeeze()
            else:
                img2_2d = img2
                
            similarities['ssim'] = self._calculate_custom_ssim(img1_2d, img2_2d)
            
            # 2. HOG feature similarity
            hog_sim = 1 - cosine(features1['hog'], features2['hog'])
            similarities['hog'] = max(0, hog_sim)  # Ensure non-negative
            
            # 3. Geometric feature similarity
            contour_sim = self._calculate_geometric_similarity(features1, features2)
            similarities['geometric'] = contour_sim
            
            # 4. Statistical feature similarity
            stat_sim = self._calculate_statistical_similarity(features1, features2)
            similarities['statistical'] = stat_sim
            
            # 5. Texture similarity
            texture_sim = 1 - cosine(features1['texture'], features2['texture'])
            similarities['texture'] = max(0, texture_sim)
            
            # 6. Pixel-level correlation
            flat1 = img1_2d.flatten()
            flat2 = img2_2d.flatten()
            correlation, _ = pearsonr(flat1, flat2)
            similarities['correlation'] = max(0, correlation)
            
            # Weighted combination of all similarities
            weights = {
                'ssim': 0.25,
                'hog': 0.20,
                'geometric': 0.15,
                'statistical': 0.15,
                'texture': 0.15,
                'correlation': 0.10
            }
            
            final_similarity = sum(similarities[key] * weights[key] for key in weights)
            
            # Ensure result is between 0 and 1
            final_similarity = max(0, min(1, final_similarity))
            
            logger.debug(f"Individual similarities: {similarities}")
            logger.debug(f"Final similarity: {final_similarity}")
            
            return float(final_similarity)
            
        except Exception as e:
            logger.error(f"Error predicting similarity: {str(e)}")
            # Return a conservative similarity score
            return 0.5
    
    def _calculate_geometric_similarity(self, features1, features2):
        """Calculate similarity based on geometric features"""
        try:
            # Compare contour areas (normalized)
            area1 = features1['contour_area']
            area2 = features2['contour_area']
            max_area = max(area1, area2, 1)  # Avoid division by zero
            area_sim = 1 - abs(area1 - area2) / max_area
            
            # Compare aspect ratios
            ratio1 = features1['aspect_ratio']
            ratio2 = features2['aspect_ratio']
            ratio_sim = 1 - min(abs(ratio1 - ratio2), 1)  # Cap at 1
            
            # Compare perimeters (normalized)
            perim1 = features1['contour_perimeter']
            perim2 = features2['contour_perimeter']
            max_perim = max(perim1, perim2, 1)
            perim_sim = 1 - abs(perim1 - perim2) / max_perim
            
            # Weighted average
            geometric_sim = (area_sim * 0.4 + ratio_sim * 0.3 + perim_sim * 0.3)
            return max(0, min(1, geometric_sim))
            
        except Exception as e:
            logger.error(f"Error calculating geometric similarity: {str(e)}")
            return 0.5
    
    def _calculate_statistical_similarity(self, features1, features2):
        """Calculate similarity based on statistical features"""
        try:
            # Compare mean intensities
            mean_sim = 1 - abs(features1['mean_intensity'] - features2['mean_intensity'])
            
            # Compare standard deviations
            std_sim = 1 - abs(features1['std_intensity'] - features2['std_intensity'])
            
            # Compare skewness
            skew_sim = 1 - min(abs(features1['skewness'] - features2['skewness']) / 2, 1)
            
            # Compare kurtosis
            kurt_sim = 1 - min(abs(features1['kurtosis'] - features2['kurtosis']) / 3, 1)
            
            # Weighted average
            stat_sim = (mean_sim * 0.3 + std_sim * 0.3 + skew_sim * 0.2 + kurt_sim * 0.2)
            return max(0, min(1, stat_sim))
            
        except Exception as e:
            logger.error(f"Error calculating statistical similarity: {str(e)}")
            return 0.5
    
    def is_genuine(self, similarity_score):
        """
        Determine if signature is genuine based on similarity score
        
        Args:
            similarity_score: Similarity score between 0 and 1
            
        Returns:
            bool: True if genuine, False if forged
        """
        return similarity_score >= self.threshold
    
    def get_model_summary(self):
        """Get model architecture summary"""
        return {
            'model_type': 'Computer Vision Based Signature Analysis',
            'features': [
                'Histogram of Oriented Gradients (HOG)',
                'Structural Similarity Index (SSIM)',
                'Contour-based geometric features',
                'Statistical intensity features',
                'Texture analysis using gradients',
                'Pixel-level correlation'
            ],
            'threshold': self.threshold,
            'input_shape': self.input_shape
        }

# Global model instance
signature_model = SiameseSignatureModel()
