import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
from load_dataset import get_dataloaders  # 🔁 Import your new loader

# === CONFIG ===
data_dir = "dataset"
model_save_path = "backend/best_model.pth"
batch_size = 32
num_epochs = 50
image_size = 224
validation_split = 0.2
learning_rate = 0.001
patience = 5  # Early stopping patience

# === LOAD DATA ===
train_loader, val_loader, class_names = get_dataloaders(
    data_dir=data_dir,
    image_size=image_size,
    batch_size=batch_size,
    validation_split=validation_split
)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# === TRAINING LOOP ===
train_losses = []
val_accuracies = []
best_val_acc = 0
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # === VALIDATION ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # === EARLY STOPPING + BEST MODEL SAVE ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        early_stop_counter = 0
        print(f" New best model saved with Val Accuracy: {val_acc:.2f}%")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

    scheduler.step()

# === PLOT ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy over Epochs")

plt.tight_layout()
plt.show()
