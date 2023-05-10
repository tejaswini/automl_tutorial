import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

from ray import tune

from automl_tutorial.utils.utils import set_seeds

def load_data(data_dir="/home/human/automl_tutorial/data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

class Net(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, conv3_filters, linear_size1):
            super(Net, self).__init__()
            self.conv3_filters = conv3_filters
            self.conv1 = nn.Conv2d(3, conv1_filters, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 5)
            self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, 5)
            self.fc1 = nn.Linear(conv3_filters * 6 * 6, linear_size1)
            self.fc2 = nn.Linear(linear_size1, 10)

    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.conv3_filters * 6 * 6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x


def train_cifar(config, num_epochs=10, checkpoint_dir=None, data_dir='/home/human/automl_tutorial/data'):
    print("config is ", config)
    set_seeds()
    net = Net(2 ** config["num_filters1"], 2 ** config["num_filters2"],
              2 ** config["num_filters3"], 2 ** config["l1"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    set_seeds()
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    set_seeds()
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=8)

    set_seeds()
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=32,
        shuffle=True,
        num_workers=8)

    for epoch in range(num_epochs):
        training_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Validation loss
        val_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()

        tune.report(loss=(val_loss / len(val_loader)), accuracy=correct / total)
    print("Finished Training")
