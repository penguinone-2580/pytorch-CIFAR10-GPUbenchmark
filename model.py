import torch
import torch.nn as nn
import torchvision

class Construct_Model:
    def __init__(self, model_name):
        if model_name=="VGG":
            self.model = torchvision.models.vgg19(pretrained=False)
        elif model_name=="ResNet":
            self.model = torchvision.models.resnet18(pretrained=False)
        else:
            print("Model name is incorrect")
            print("Set model to VGG19")
            self.model = torchvision.models.vgg19(pretrained=False)
