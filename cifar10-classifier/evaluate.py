import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from train import BaselineCNN, ImprovedResNet

def evaluate_model(model, test_loader, device, name):
    model.load_state_dict(torch.load(f'models/{name}.pth'))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'\n=== {name.upper()} Results ===')
    print(f'Test Accuracy: {acc*100:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{name.capitalize()} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{name}_confusion_matrix.png', dpi=150)
    print(f'Confusion matrix saved to results/{name}_confusion_matrix.png')
    
    return acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    baseline_acc = evaluate_model(BaselineCNN(), test_loader, device, 'baseline')
    improved_acc = evaluate_model(ImprovedResNet(), test_loader, device, 'improved')
    
    print(f'\n=== COMPARISON ===')
    print(f'Baseline: {baseline_acc*100:.2f}%')
    print(f'Improved: {improved_acc*100:.2f}%')
    print(f'Improvement: +{(improved_acc-baseline_acc)*100:.2f}%')
