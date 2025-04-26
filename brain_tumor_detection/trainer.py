import os
import ssl
import copy
import pandas as pd
import matplotlib.pyplot as plt

# SSL bypass for macOS cert issues
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# === Configuration ===
dataset_dir = "./brain_tumor_dataset"
batch_size = 32
img_size = 224
validation_split = 0.2
seed = 123
num_epochs = 30
learning_rate = 1e-3
num_workers = 4  # Adjust based on your CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)

# Define experiment variations
loss_functions = {
    'BCE': nn.BCEWithLogitsLoss(),
    'MSE': nn.MSELoss()
}
optimizer_builders = {
    'Adam': lambda params: optim.Adam(params, lr=learning_rate),
    'SGD': lambda params: optim.SGD(params, lr=learning_rate, momentum=0.9)
}
scheduler_builders = {
    'None': None,
    'ReduceLROnPlateau': lambda opt: ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=3),
    'StepLR': lambda opt: StepLR(opt, step_size=7, gamma=0.1)
}
model_builders = {
    'simple_cnn': lambda: nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*(img_size//8)**2,128), nn.ReLU(),
        nn.Linear(128,1)
    ),
    'vgg16': lambda: (_model_factory := models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
                      [_param.requires_grad_(False) for _param in _model_factory.features],
                      setattr(_model_factory.classifier, '6', nn.Linear(4096,1)),
                      _model_factory)[-1],
    'resnet50': lambda: (_model_factory := models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
                         [_param.requires_grad_(False) for _param in _model_factory.parameters()],
                         setattr(_model_factory, 'fc', nn.Linear(_model_factory.fc.in_features,1)),
                         _model_factory)[-1]
}

# Data loaders
def prepare_data_loaders():
    transforms_map = {
        'train': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }
    dataset = datasets.ImageFolder(dataset_dir, transform=transforms_map['train'])
    total = len(dataset)
    val_count = int(total * validation_split)
    train_count = total - val_count
    train_ds, val_ds = random_split(dataset, [train_count, val_count])
    val_ds.dataset.transform = transforms_map['val']

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    sizes = {'train': train_count, 'val': val_count}
    return loaders, sizes

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop
def train_model(model, loaders, sizes, criterion, optimizer, scheduler, epochs):
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    stopper = EarlyStopping(patience=5)

    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            running_loss = running_corrects = 0
            for imgs, labels in loaders[phase]:
                imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs)>0.5).float()
                    if phase=='train':
                        loss.backward(); optimizer.step()
                running_loss += loss.item()*imgs.size(0)
                running_corrects += torch.sum(preds==labels)
            epoch_loss = running_loss/sizes[phase]
            epoch_acc = (running_corrects.double()/sizes[phase]).item()
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase=='val':
                if scheduler: scheduler.step(epoch_loss)
                stopper(epoch_loss)
                if epoch_acc>best_acc:
                    best_acc=epoch_acc; best_wts=copy.deepcopy(model.state_dict())
        if stopper.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(best_wts)
    return model, history

# Main experiment
if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()

    loaders, sizes = prepare_data_loaders()
    results = []

    for model_name, build_model in model_builders.items():
        for loss_name, criterion in loss_functions.items():
            for opt_name, opt_fn in optimizer_builders.items():
                model = build_model()
                optimizer = opt_fn(filter(lambda p: p.requires_grad, model.parameters()))
                for sched_name, sched_builder in scheduler_builders.items():
                    scheduler = sched_builder(optimizer) if sched_builder else None
                    tag = f"{model_name}_{loss_name}_{opt_name}_{sched_name}"
                    print(f"\nExperiment: {tag}")
                    _, hist = train_model(model, loaders, sizes, criterion, optimizer, scheduler, num_epochs)
                    best_epoch = hist['val_acc'].index(max(hist['val_acc']))+1
                    best_acc = max(hist['val_acc'])
                    results.append({
                        'experiment': tag,
                        'model': model_name,
                        'loss': loss_name,
                        'optimizer': opt_name,
                        'scheduler': sched_name,
                        'best_epoch': best_epoch,
                        'best_val_accuracy': best_acc
                    })

    df = pd.DataFrame(results)
    print(df)
    df.to_csv('hyperparam_experiments.csv', index=False)

    # Plot top 5 experiments
    top5 = df.nlargest(5, 'best_val_accuracy')
    plt.figure();
    for tag in top5['experiment']:
        exp = next(r for r in results if r['experiment']==tag)
        hist = exp and exp
    # (Plotting histories omitted; review CSV)

    print("All experiments complete. See 'hyperparam_experiments.csv' for details.")