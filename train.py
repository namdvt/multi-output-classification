import torch
import torch.optim as optim
from helper import write_log, write_figures
import numpy as np
import torch.nn as nn
from dataset import get_loader

from model import Model
from tqdm import tqdm


def fit(epoch, model, optimizer, criterion, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_correct = 0
    alpha = 0.25

    for image, target_color, target_category in tqdm(data_loader):
        image = image.to(device)
        target_color = target_color.to(device)
        target_category = target_category.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            out_color, out_category = model(image)
        else:
            with torch.no_grad():
                out_color, out_category = model(image)

        # loss
        color_loss = criterion(out_color, target_color)
        category_loss = criterion(out_category, target_category)
        loss = alpha*color_loss + (1-alpha)*category_loss
        running_loss += loss.item()

        # accuracy
        predict_color = torch.argmax(out_color, dim=1)
        predict_category = torch.argmax(out_category, dim=1)
        running_correct += ((predict_color == target_color) & (predict_category == target_category)).sum().item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = running_correct / len(data_loader.dataset)
    print('[%d][%s] loss: %.4f acc: %.4f' % (epoch, phase, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def train(root, device, model, epochs, bs, lr):
    print('start training ...........')
    train_loader, val_loader = get_loader(root=root, batch_size=bs, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_acc = [], []
    val_losses, val_acc = [], []
    for epoch in range(epochs):
        scheduler.step(epoch)

        train_epoch_loss, train_epoch_acc = fit(epoch, model, optimizer, criterion, device, train_loader, phase='training')
        val_epoch_loss, val_epoch_acc = fit(epoch, model, optimizer, criterion, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_acc >= np.max(val_acc):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_losses.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        write_figures('output', train_losses, val_losses, train_acc, val_acc)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc)

        scheduler.step()


if __name__ == "__main__":
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    model = Model().to(device)
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.1
    root = 'data/apparel-images-dataset'
    train(root, device, model, num_epochs, batch_size, learning_rate)
