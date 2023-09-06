import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import model, criterion

mnist_transform = transforms.ToTensor()

test_dataset = datasets.MNIST('test/', train=False, download=True, transform=mnist_transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

model.load_state_dict(torch.load('cnn-v1'))

losses = []
accuracy = []

if __name__ == '__main__':
    model.eval()

    with torch.no_grad():
        for _ in range(1):
            for X, y in iter(test_loader):
                logits = model(X)
                loss = criterion(logits, y)

                losses.append(loss.item())

                acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
                accuracy.append(acc)

                print('Loss:', loss.item(), 'Accuracy:', acc)

        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracy) / len(accuracy)
        print('Average loss:', avg_loss)
        print('Average accuracy:', avg_accuracy)
