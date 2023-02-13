import torch
from torch import datasets
import torchvision.transforms as transforms

def load_cifar10(path, batch_size=128):
    """ Loads the CIFAR10 dataset and save images in path
    """
    cuda = "cuda"
    train_dataset = datasets.CIFAR10(path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
        transforms.RandomHorizontalFlip(),
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=1)
    
    test_dataset = datasets.CIFAR10(path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=1)

    return train_loader, test_loader


def load_caltech101(path, batch_size=128, img_size=128):
    # Donwload the dataset
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.Caltech101(path, download=True, transform=transform)

    # Train test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,lengths=[0.7,0.3],generator=torch.Generator().manual_seed(42))

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

    return train_loader, test_loader