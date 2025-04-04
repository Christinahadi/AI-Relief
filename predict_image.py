import os
import torch
from torchvision import models
from PIL import Image
from torchvision.models import ResNet18_Weights
from utils import default_transform, class_names, get_topk_probs

# === Configuration ===
model_path = "backend/damage_model.pth"
image_path = "test/Destroyed102.jpg"  # <--- Change this to your local image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Predict ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = default_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)[0]

    top_predictions = get_topk_probs(logits, k=3)
    probs = torch.nn.functional.softmax(logits, dim=0)

    print("\n Top Prediction:")
    print(f"  {top_predictions[0][0]} ({top_predictions[0][1]}%)\n")

    print(" Top 3 Predictions:")
    for label, prob in top_predictions:
        print(f"- {label}: {prob}%")

    print("\n Softmax Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"- {cls}: {probs[i] * 100:.2f}%")

    print("\n Raw Logits:")
    for i, cls in enumerate(class_names):
        print(f"- {cls}: {logits[i].item():.4f}")

# === Run prediction ===
if __name__ == "__main__":
    if not os.path.exists(image_path):
        print(f" Image not found at: {image_path}")
    else:
        predict_image(image_path)
