"""
Automated evaluation script with MLOps metrics and model versioning.
Evaluates the trained model on validation set and tracks performance metrics.
"""

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np


class ModelEvaluator:
    """Evaluate model performance and track metrics."""
    
    def __init__(self, model_path='cat_presence_classifier.pth', 
                 data_dir='./labeled-frames', metrics_dir='./model_metrics'):
        """
        Initialize evaluator.
        
        Args:
            model_path (str): Path to saved model weights
            data_dir (str): Path to labeled frames directory
            metrics_dir (str): Directory to save metrics and reports
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.metrics_dir = metrics_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['cat-present', 'no-cat']
        
        # Create metrics directory if it doesn't exist
        Path(self.metrics_dir).mkdir(exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Define validation transforms (same as training validation)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the trained model."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate(self):
        """
        Evaluate model on validation set and compute metrics.
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Load validation dataset
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'validation'),
            self.transform
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        all_confidences = []
        
        print("Running evaluation on validation set...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(probs.max(dim=1)[0].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # Compute metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'accuracy': float(accuracy_score(all_labels, all_preds)),
            'precision': float(precision_score(all_labels, all_preds, average='weighted')),
            'recall': float(recall_score(all_labels, all_preds, average='weighted')),
            'f1': float(f1_score(all_labels, all_preds, average='weighted')),
            'num_samples': len(all_labels),
            'mean_confidence': float(np.mean(all_confidences)),
            'min_confidence': float(np.min(all_confidences)),
            'max_confidence': float(np.max(all_confidences)),
        }
        
        # Per-class metrics
        metrics['per_class_report'] = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics, all_preds, all_labels, all_confidences
    
    def save_metrics(self, metrics):
        """Save metrics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_file}")
        return metrics_file
    
    def print_metrics(self, metrics):
        """Print metrics in a readable format."""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"\nSamples evaluated: {metrics['num_samples']}")
        print(f"Mean confidence:   {metrics['mean_confidence']:.4f}")
        print(f"Min/Max confidence: {metrics['min_confidence']:.4f} / {metrics['max_confidence']:.4f}")
        
        print("\n" + "-"*50)
        print("Per-class metrics:")
        print("-"*50)
        for class_name in self.class_names:
            class_metrics = metrics['per_class_report'][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1-Score:  {class_metrics['f1-score']:.4f}")
    
    def plot_confusion_matrix(self, all_preds, all_labels, metrics_file_prefix):
        """Create and save confusion matrix visualization."""
        cm = confusion_matrix(all_labels, all_preds)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plot_path = os.path.join(self.metrics_dir, f"confusion_matrix_{metrics_file_prefix}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {plot_path}")
        plt.close()
    
    def plot_confidence_distribution(self, all_confidences, metrics_file_prefix):
        """Create and save confidence distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_confidences, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Model Confidence Scores')
        ax.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_confidences):.3f}')
        ax.legend()
        
        fig.tight_layout()
        plot_path = os.path.join(self.metrics_dir, f"confidence_dist_{metrics_file_prefix}.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Confidence distribution saved to {plot_path}")
        plt.close()


def main():
    """Run full evaluation pipeline."""
    evaluator = ModelEvaluator()
    
    # Evaluate model
    metrics, all_preds, all_labels, all_confidences = evaluator.evaluate()
    
    # Print results
    evaluator.print_metrics(metrics)
    
    # Save metrics
    metrics_file = evaluator.save_metrics(metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(all_preds, all_labels, timestamp)
    evaluator.plot_confidence_distribution(all_confidences, timestamp)
    
    print(f"\n✓ Evaluation complete! Check {evaluator.metrics_dir} for results.")


if __name__ == "__main__":
    main()
