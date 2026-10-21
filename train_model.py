# Train pytorch model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import os
import time
import copy
import warnings
from pathlib import Path

# Suppress harmless UserWarnings from Pillow or torchvision, if any
warnings.filterwarnings("ignore", category=UserWarning)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    """
    Trains a PyTorch model and evaluates it on a validation set.

    Args:
        model: The PyTorch model to train.
        dataloaders (dict): Dictionary of DataLoaders for 'train' and 'validation' phases.
        dataset_sizes (dict): Dictionary of dataset sizes for 'train' and 'validation' phases.
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The optimization algorithm (e.g., optim.Adam).
        num_epochs (int): The number of training epochs.

    Returns:
        model: The trained model with the best validation accuracy.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Ensure device is set (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # Move model to the selected device

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # Move inputs to device
                labels = labels.to(device) # Move labels to device

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the predicted class
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() # Newline for better readability between epochs

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / 'labeled-frames'
    model_save_path = Path(__file__).resolve().parent / 'cat_presence_classifier.pth'

    # --- 1. Define Data Transformations ---
    # These are standard transformations for pre-trained models like ResNet
    # ImageNet normalization values are used as the pre-trained model was trained on ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(), # Randomly flip images horizontally for augmentation
            transforms.ToTensor(),             # Convert PIL Image or NumPy array to PyTorch Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize with ImageNet stats
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),            # Resize the shorter side to 256
            transforms.CenterCrop(224),        # Crop the central 224x224 portion
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading data from: {data_dir}")

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), data_transforms['train']
    )
    validation_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'validation'), data_transforms['validation']
    )
    if train_dataset.classes != ['cat-present', 'no-cat']:
        raise ValueError(
            f"Expected class order ['cat-present', 'no-cat'], got {train_dataset.classes}."
        )
    if validation_dataset.classes != train_dataset.classes:
        raise ValueError("Training and validation folders must use the same classes.")

    train_clip_sources = {
        Path(path).stem.rsplit('_', 1)[0]
        for path, _ in train_dataset.samples
    }
    validation_indices = [
        index
        for index, (path, _) in enumerate(validation_dataset.samples)
        if Path(path).stem.rsplit('_', 1)[0] not in train_clip_sources
    ]
    if not validation_indices:
        raise ValueError(
            "No clip-disjoint validation images remain. Add validation clips "
            "that are not also present in the training set."
        )
    validation_sources = {
        Path(validation_dataset.samples[index][0]).stem.rsplit('_', 1)[0]
        for index in validation_indices
    }
    validation_subset = Subset(validation_dataset, validation_indices)
    print(
        f"Using {len(validation_indices)} validation frames from "
        f"{len(validation_sources)} clips not seen in training."
    )

    # DataLoaders for batching and shuffling
    dataloaders = {
        'train': DataLoader(
            train_dataset, batch_size=32, shuffle=True,
            num_workers=os.cpu_count() // 2 or 1
        ),
        'validation': DataLoader(
            validation_subset, batch_size=32, shuffle=False, num_workers=0
        ),
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'validation': len(validation_subset),
    }
    print(f"Training dataset size: {dataset_sizes['train']}")
    print(f"Clip-disjoint validation dataset size: {dataset_sizes['validation']}")

    # --- 3. Determine Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Continue fine-tuning the project's existing checkpoint instead of
    # replacing it with a generic pretrained model.
    if not model_save_path.is_file():
        raise FileNotFoundError(
            f"Existing model checkpoint not found: {model_save_path}"
        )
    model_ft = models.resnet18(weights=None)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model_ft.load_state_dict(torch.load(model_save_path, map_location=device))

    # --- 5. Define Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001)

    # --- 6. Train the Model ---
    print("\nStarting model training...")
    model_ft = train_model(
        model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft,
        num_epochs=5
    )

    # --- 7. Save the Trained Model ---
    torch.save(model_ft.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")
