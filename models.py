import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VerificationResult:
    """Model to represent signature verification results"""
    
    def __init__(self, genuine_image_path, test_image_path, similarity_score, is_genuine, threshold=0.7):
        self.genuine_image_path = genuine_image_path
        self.test_image_path = test_image_path
        self.similarity_score = similarity_score
        self.is_genuine = is_genuine
        self.threshold = threshold
        self.timestamp = datetime.now()
        
    def to_dict(self):
        """Convert result to dictionary for JSON response"""
        return {
            'genuine_image': self.genuine_image_path,
            'test_image': self.test_image_path,
            'similarity_score': float(self.similarity_score),
            'is_genuine': self.is_genuine,
            'verdict': 'Genuine' if self.is_genuine else 'Forged',
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat()
        }
    
    def get_confidence_level(self):
        """Get confidence level description based on similarity score"""
        if self.similarity_score >= 0.9:
            return 'Very High'
        elif self.similarity_score >= 0.8:
            return 'High'
        elif self.similarity_score >= 0.7:
            return 'Medium'
        elif self.similarity_score >= 0.6:
            return 'Low'
        else:
            return 'Very Low'
    
    def get_verdict_class(self):
        """Get CSS class for verdict styling"""
        return 'text-success' if self.is_genuine else 'text-danger'
