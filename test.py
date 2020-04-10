from dataset import ApparelImagesDataset, get_loader
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from PIL import Image
import torchvision.transforms as tf
import torch
from model import Model
import os

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

color_classes = ('black', 'blue', 'brown', 'green', 'red', 'white')
idx_to_color = dict(enumerate(color_classes))
color_to_idx = {char: idx for idx, char in idx_to_color.items()}

category_classes = ('dress', 'pants', 'shirt', 'shoes', 'shorts')
idx_to_category = dict(enumerate(category_classes))
category_to_idx = {char: idx for idx, char in idx_to_category.items()}


transforms = tf.Compose([
    tf.Resize([100, 100]),
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict(file):
    image = Image.open(file)
    image = transforms(image).unsqueeze(0).to(device)

    model = Model().to(device)
    model.load_state_dict(torch.load('output/weight.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        out_color, out_category = model(image)
    color_idx = torch.argmax(out_color, dim=1)
    category_idx = torch.argmax(out_category, dim=1)

    print(idx_to_color[color_idx.item()] + '_' + idx_to_category[category_idx.item()])
    
    
if __name__ == '__main__':
    predict('test_images/giay_trang.jpg')
