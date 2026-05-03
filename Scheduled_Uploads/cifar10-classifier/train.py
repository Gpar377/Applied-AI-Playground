import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# Baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# Improved ResNet
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)) if stride != 1 or in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return nn.ReLU()(self.conv(x) + self.skip(x))

class ImprovedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def train_model(model, train_loader, val_loader, epochs, lr, device, name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        print(f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'models/{name}.pth')
    
    return best_acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_data, val_data = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    # Baseline
    print('\n=== Training Baseline CNN ===')
    baseline = BaselineCNN()
    baseline_acc = train_model(baseline, train_loader, val_loader, 30, 0.001, device, 'baseline')
    print(f'Baseline Best Val Acc: {baseline_acc:.2f}%')
    
    # Improved
    print('\n=== Training Improved ResNet ===')
    improved = ImprovedResNet()
    improved_acc = train_model(improved, train_loader, val_loader, 50, 0.01, device, 'improved')
    print(f'Improved Best Val Acc: {improved_acc:.2f}%')
