import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from model import model

model.load_state_dict(torch.load('cnn-v1'))

IMG1_PATH = 'img/img_27311.jpg'
IMG2_PATH = 'img/img_27355.jpg'

img1 = Image.open(IMG1_PATH)
img2 = Image.open(IMG2_PATH)

img1_tensor = pil_to_tensor(img1).float()
img2_tensor = pil_to_tensor(img2).float()

cosim = nn.CosineSimilarity()

def get_embed(X):
    model.eval()
    with torch.no_grad():
        return model.cnn(X)

if __name__ == '__main__':
    emb1 = get_embed(img1_tensor.unsqueeze(0))
    emb2 = get_embed(img2_tensor.unsqueeze(0))

    similarity = cosim(emb1, emb2)

    print('Similarity:', similarity.item())
