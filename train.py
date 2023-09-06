import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import model, optimizer, criterion

mnist_transform = transforms.ToTensor()

train_dataset = datasets.MNIST('data/', download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=32)

losses = []
accuracy = []

if __name__ == '__main__':
    model.train()

    for _ in range(1):
        for X, y in iter(train_loader):
            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)
            accuracy.append(acc)

            print('Loss:', loss.item(), 'Accuracy:', acc)



    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracy) / len(accuracy)
    print('Average loss:', avg_loss)
    print('Average accuracy:', avg_accuracy)

    torch.save(model.state_dict(), 'cnn-v1')
