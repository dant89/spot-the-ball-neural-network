import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import os
import re

from neural_network import Net


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class SpotTheBallDataset(Dataset):
    def __init__(self, directory, transform=None, max_width=5000, max_height=4000):
        self.directory = directory
        self.transform = transform
        self.max_width = max_width
        self.max_height = max_height
        self.images = [img for img in os.listdir(directory) if img.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        match = re.search(r'--(\d+)-(\d+).png', self.images[idx])
        if match:
            x_variable, y_variable = map(int, match.groups())
            x_variable /= self.max_width
            y_variable /= self.max_height
        else:
            x_variable, y_variable = 0.0, 0.0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([x_variable, y_variable], dtype=torch.float)


class Trainer:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SpotTheBallDataset('images_training/', transform=transform)
    trainloader = DataLoader(dataset, batch_size=2, shuffle=True)

    net = Net()
    net.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training the model
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = 'spot_the_ball_model.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    Trainer()
