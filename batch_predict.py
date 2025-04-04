import os
import torch
import pandas as pd
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
from utils import default_transform, class_names, get_topk_probs

# === Config ===
model_path = "backend/damage_model.pth"
image_folder = "test"  # Change this to your folder path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === Storage for CSV Output ===
results = []

# === Run Predictions ===
print(f"\n Scanning folder: {image_folder}\n")
for filename in os.listdir(image_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    input_tensor = default_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)[0]

    top_predictions = get_topk_probs(logits, k=3)

    print(f" {filename}")
    print(f"   Top Prediction: {top_predictions[0][0]} ({top_predictions[0][1]}%)")
    print("   Top 3:")
    for label, prob in top_predictions:
        print(f"    - {label}: {prob}%")
    print()

    # Append to CSV results
    results.append({
        "filename": filename,
        "top_class": top_predictions[0][0],
        "top_confidence": top_predictions[0][1],
        "top_1": top_predictions[0][0],
        "conf_1": top_predictions[0][1],
        "top_2": top_predictions[1][0],
        "conf_2": top_predictions[1][1],
        "top_3": top_predictions[2][0],
        "conf_3": top_predictions[2][1],
    })

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("batch_predictions.csv", index=False)
print(" Results saved to: batch_predictions.csv")
