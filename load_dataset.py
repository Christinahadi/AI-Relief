from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir="dataset", image_size=224, batch_size=32, validation_split=0.2):
    """
    Loads image data from the specified directory and returns training and validation DataLoaders.

    Parameters:
    - data_dir (str): Path to dataset folder with class subdirectories.
    - image_size (int): Target size to resize all images (default: 224x224).
    - batch_size (int): Number of samples per batch.
    - validation_split (float): Fraction of data to use for validation (e.g., 0.2 = 20%).

    Returns:
    - train_loader (DataLoader): DataLoader for training data
    - val_loader (DataLoader): DataLoader for validation data
    - class_names (List[str]): List of class labels
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with class labels from folder names
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes

    # Split dataset into training and validation sets
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, class_names
