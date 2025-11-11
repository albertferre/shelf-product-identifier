import torch

from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class Img2VecDino2():
    def __init__(self):
        # Set the device to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Specify the model name as "resnet-18"
        self.modelName = "facebook/dinov2-base"

        self.processor = AutoImageProcessor.from_pretrained(self.modelName)
        # Move the model to the device
        self.model = AutoModel.from_pretrained(self.modelName).to(self.device)

    def getVec(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze().cpu().numpy()
