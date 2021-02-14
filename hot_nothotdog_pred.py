import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
mean = (0.6624, 0.5741, 0.4946)
stdev = (0.2218, 0.2427, 0.2639)
tfms = tt.Compose([
    tt.Resize((299, 299)),
    tt.ToTensor(),
    tt.Normalize(mean, stdev, inplace=True)
])
# Functions to move data to GPU if available


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()


def accuracy(outputs, labels):
    preds = torch.max(outputs, dim=1)[1]
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    # Training step takes in batch of data and returns loss for that batch
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        # Using cross_entropy as loss_fcn
        return F.cross_entropy(out, labels)
    # Validation step is basically a repeat of the training step

    def val_step(self, batch):
        with torch.no_grad():
            images, labels = batch
            out = self(images)
            print(f"Labels: {labels}; predictions: {out}")
            # Use cross_entropy loss function because this is a classification problem
            loss = F.cross_entropy(out, labels)
            acc = accuracy(out, labels)
        return loss.detach(), acc

    def pred_img(self, img):
        with torch.no_grad():
            return self(img)
    # Add loss and accuracy and return the values in the form of tuple

    def val_anal(self, loss_acc):
        batch_loss = [loss[0] for loss in loss_acc]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [acc[1] for acc in loss_acc]
        epoch_acc = torch.stack(batch_acc).mean()
        return epoch_loss, epoch_acc
    # Print loss and accuracy for given epoch

    def print_anal(self, epoch, result):
        print(f"----------------Epoch: {epoch}----------------")
        print(
            f"Last learning rate: {result[3][-1]}\nEpoch loss(val): {result[0]}\nEpoch loss(train): {result[2]}\nEpoch accuracy: {result[1]}\n")

# Define convolutional block with convolutional layer, batch normalization, activation function, and optional pooling


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def res_block(chan):
    layers = [
        nn.Conv2d(chan, chan, kernel_size=3, padding=1),
        nn.BatchNorm2d(chan),
        nn.ReLU(inplace=True),
        nn.Conv2d(chan, chan, kernel_size=3, padding=1),
        nn.BatchNorm2d(chan),
        nn.ReLU(inplace=True),
        nn.Conv2d(chan, chan, kernel_size=3, padding=1),
        nn.BatchNorm2d(chan),
        nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Resnet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64, pool=True)  # 64x64x64
        self.layer1 = res_block(64)
        self.conv2 = conv_block(64, 128, pool=True)  # 128x32x32
        self.layer2 = res_block(128)
        self.conv3 = conv_block(128, 256, pool=True)  # 256x16x16
        self.layer3 = res_block(256)
        self.conv4 = conv_block(256, 512, pool=True)  # 512x8x8
        self.layer4 = res_block(512)
        self.conv5 = conv_block(512, 1024, pool=True)  # 1024x4x4
        self.layer5 = res_block(1024)
        self.conv6 = conv_block(1024, 1024, pool=True)
        self.layer6 = res_block(1024)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(1024, 2)
        ])

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.layer1(xb) + xb
        xb = self.conv2(xb)
        xb = self.layer2(xb) + xb
        xb = self.conv3(xb)
        xb = self.layer3(xb) + xb
        xb = self.conv4(xb)
        xb = self.layer4(xb) + xb
        xb = self.conv5(xb)
        xb = self.layer5(xb) + xb
        xb = self.conv6(xb)
        xb = self.layer6(xb) + xb
        xb = self.avgpool(xb)
        return self.fc(xb)


@torch.no_grad()
def predict_image(xb, model_path):
    model = to_device(Resnet(3, 2), device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    xb = xb.convert('RGB')
    xb = tfms(xb).unsqueeze(0)
    xb = to_device(xb, device)
    # IMportant to include model.eval()
    model.eval()
    yb = model.pred_img(xb)
    print(yb)
    return torch.max(yb, dim=1)[1]
