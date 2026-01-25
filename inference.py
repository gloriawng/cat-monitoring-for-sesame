"""
Inference module for cat presence classifier.
Provides functions to load the model and make predictions on images.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path


class CatPresenceClassifier:
    """Load and use the trained cat presence classifier model."""
    
    def __init__(self, model_path='cat_presence_classifier.pth', device=None):
        """
        Initialize the classifier.
        
        Args:
            model_path (str): Path to the saved model weights
            device (torch.device): Device to run inference on (defaults to GPU if available)
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['cat-present', 'no-cat']
        
        # Load model architecture
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on device: {self.device}")
    
    def predict(self, image_path):
        """
        Predict on a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: {'class': str, 'label': int, 'confidence': float, 'probabilities': dict}
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {'error': f"Could not load image: {e}"}
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        return {
            'class': self.class_names[predicted_label],
            'label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.class_names))
            }
        }
    
    def predict_folder(self, folder_path):
        """
        Predict on all images in a folder.
        
        Args:
            folder_path (str): Path to folder with images
            
        Returns:
            list: List of predictions with file paths
        """
        results = []
        folder = Path(folder_path)
        image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        for i, image_path in enumerate(image_files):
            prediction = self.predict(str(image_path))
            prediction['file'] = str(image_path)
            results.append(prediction)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(image_files)}")
        
        return results


if __name__ == "__main__":
    import sys
    
    # Example usage: python inference.py <image_path>
    if len(sys.argv) > 1:
        classifier = CatPresenceClassifier()
        result = classifier.predict(sys.argv[1])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPrediction: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"All probabilities: {result['probabilities']}")
    else:
        print("Usage: python inference.py <image_path>")
        print("\nExample:")
        print("  python inference.py labeled-frames/train/cat-present/image.jpg")
