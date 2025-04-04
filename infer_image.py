import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import gradio as gr
from utils import default_transform, class_names, get_topk_probs

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("backend/damage_model.pth", map_location=device))
model = model.to(device)
model.eval()

# === Prediction Function ===
def predict_image_gradio(image):
    image = image.convert("RGB")
    image_tensor = default_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)[0]

    top_preds = get_topk_probs(logits, k=3)
    label, confidence = top_preds[0]

    result = f"Predicted: {label} ({confidence}%)\n\nTop 3:\n"
    for lbl, prob in top_preds:
        result += f"{lbl}: {prob}%\n"
    return result

# === Gradio UI ===
gr.Interface(
    fn=predict_image_gradio,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI Relief: Damage Classification",
    description="Upload an image of a disaster-affected area to detect the level of damage."
).launch()