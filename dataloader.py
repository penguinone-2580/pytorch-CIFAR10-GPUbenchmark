import os
import torch
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, dataset_path, batch_size, num_workers=0, pin_memory=True):
        transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(3),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        transform_val = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_val)

        if num_workers == 0:
            num_workers = os.cpu_count()
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
