import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the model architecture
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: cat-present, no-cat

# Load the trained weights
model.load_state_dict(torch.load('cat_presence_classifier.pth'))
model = model.to(device)
model.eval()

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict on an image
image = Image.open(
    'labeled-frames/train/no-cat/frame_44-cat-bowl-2025-04-23t11-37-50-00-00_00001.jpg'
)
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

print(f"Class: {['cat-present', 'no-cat'][predicted_class]}, Confidence: {confidence:.2%}")