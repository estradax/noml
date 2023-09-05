import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import model, criterion

mnist_transform = transforms.ToTensor()

test_dataset = datasets.MNIST('test/', train=False, download=True, transform=mnist_transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

model.load_state_dict(torch.load('cnn-v1'))

losses = []

if __name__ == '__main__':
    model.eval()

    with torch.no_grad():
        for _ in range(1):
            for X, y in iter(test_loader):
                logits = model(X)
                loss = criterion(logits, y)

                losses.append(loss.item())
                print('Loss:', loss.item())

        avg_loss = sum(losses) / len(losses)
        print('Average loss:', avg_loss)
