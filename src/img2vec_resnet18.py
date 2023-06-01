import torch
from tqdm import tqdm
from torchvision import models
from torchvision import transforms

class Img2VecResnet18():
    def __init__(self):
        # Set the device to CPU
        self.device = torch.device("cpu")
        # Define the number of features extracted by the model
        self.numberFeatures = 512
        # Specify the model name as "resnet-18"
        self.modelName = "resnet-18"
        # Get the model and feature layer
        self.model, self.featureLayer = self.getFeatureLayer()
        # Move the model to the device
        self.model = self.model.to(self.device)
        # Set the model to evaluation mode
        self.model.eval()
        # Initialize the transformation to convert images to tensors
        self.toTensor = transforms.ToTensor()
        # Initialize the normalization transformation for image tensors
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        # Convert the image to a tensor and normalize it
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        # Create a tensor for storing the feature embedding
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o):
            # Function to copy the output tensor to the embedding tensor
            embedding.copy_(o.data)

        # Register a forward hook to the feature layer
        h = self.featureLayer.register_forward_hook(copyData)
        # Pass the image through the model
        self.model(image)
        # Remove the forward hook
        h.remove()

        # Convert the embedding tensor to a numpy array and return it
        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        # Create an instance of the ResNet-18 model
        cnnModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Retrieve the average pooling layer (feature layer) from the model
        layer = cnnModel._modules.get('avgpool')
        # Set the output size of the feature layer
        self.layer_output_size = 512

        return cnnModel, layer
