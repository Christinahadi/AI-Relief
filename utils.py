from torchvision import transforms
import torch.nn.functional as F

# === Standard image transform used across all scripts ===
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Class labels in the order used during training ===
class_names = ["Affected", "Destroyed", "Major", "Minor", "NoDamage"]

# === Get top-k softmax probabilities with labels ===
def get_topk_probs(logits, k=3):
    """
    Returns top-k (label, confidence) pairs from model logits.
    
    Args:
        logits (Tensor): Raw output from the model.
        k (int): Number of top classes to return.

    Returns:
        List[Tuple[str, float]]: [(label, confidence%), ...]
    """
    probs = F.softmax(logits, dim=0)
    topk = probs.topk(k)
    return [(class_names[i], round(topk.values[j].item() * 100, 2)) for j, i in enumerate(topk.indices)]
