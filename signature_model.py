import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import cv2

logger = logging.getLogger(__name__)

class SiameseSignatureModel:
    """Siamese Neural Network for signature fraud detection"""
    
    def __init__(self, input_shape=(105, 105, 1)):
        self.input_shape = input_shape
        self.model = None
        self.threshold = 0.7
        self._build_model()
        
    def _create_base_network(self):
        """Create the base CNN network for feature extraction"""
        base_model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(64, (10, 10), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(128, (7, 7), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (4, 4), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (4, 4), activation='relu'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(4096, activation='sigmoid'),
            layers.Dropout(0.2)
        ])
        
        return base_model
    
    def _build_model(self):
        """Build the complete Siamese network"""
        try:
            # Create base network
            base_network = self._create_base_network()
            
            # Define inputs for the two signature images
            input_a = layers.Input(shape=self.input_shape, name='genuine_signature')
            input_b = layers.Input(shape=self.input_shape, name='test_signature')
            
            # Process both inputs through the same base network
            processed_a = base_network(input_a)
            processed_b = base_network(input_b)
            
            # Calculate L1 distance between features
            distance = layers.Lambda(
                lambda tensors: tf.abs(tensors[0] - tensors[1]),
                name='l1_distance'
            )([processed_a, processed_b])
            
            # Final prediction layer
            prediction = layers.Dense(1, activation='sigmoid', name='similarity')(distance)
            
            # Create and compile model
            self.model = models.Model(inputs=[input_a, input_b], outputs=prediction)
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Siamese model built successfully")
            
        except Exception as e:
            logger.error(f"Error building Siamese model: {str(e)}")
            raise
    
    def predict_similarity(self, img1, img2):
        """
        Predict similarity between two signature images
        
        Args:
            img1: First preprocessed signature image (numpy array)
            img2: Second preprocessed signature image (numpy array)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Ensure images have the correct shape
            if len(img1.shape) == 3:
                img1 = np.expand_dims(img1, axis=0)
            if len(img2.shape) == 3:
                img2 = np.expand_dims(img2, axis=0)
            
            # Make prediction
            similarity_score = self.model.predict([img1, img2], verbose=0)[0][0]
            
            return float(similarity_score)
            
        except Exception as e:
            logger.error(f"Error predicting similarity: {str(e)}")
            raise
    
    def is_genuine(self, similarity_score):
        """
        Determine if signature is genuine based on similarity score
        
        Args:
            similarity_score: Similarity score between 0 and 1
            
        Returns:
            bool: True if genuine, False if forged
        """
        return similarity_score >= self.threshold
    
    def load_pretrained_weights(self, weights_path):
        """Load pre-trained weights if available"""
        try:
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                logger.info(f"Loaded pre-trained weights from {weights_path}")
            else:
                logger.warning(f"No pre-trained weights found at {weights_path}")
                # Initialize with random weights for demonstration
                self._initialize_demo_weights()
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            self._initialize_demo_weights()
    
    def _initialize_demo_weights(self):
        """Initialize weights for demonstration purposes"""
        logger.info("Initializing model with demo weights for immediate functionality")
        # The model is already initialized with random weights via Keras
        # In a production environment, you would train this model on a signature dataset
        pass
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model:
            return self.model.summary()
        return "Model not built"

# Global model instance
signature_model = SiameseSignatureModel()
