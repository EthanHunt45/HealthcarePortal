import os
import ssl
import copy
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------------------
# Configuration
# ---------------------
SOURCE_ROOT = "C:/Users/bahar/PycharmProjects/Mammography/minidatasets/IDC_regular_ps50_idx5"
TARGET_ROOT = "datasets/processed_dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 30
LR = 1e-3
SEED = 42
PATIENCE = 5

torch.manual_seed(SEED)

# ---------------------
# Data Preparation
# ---------------------
def collect_images():
    for label in ["0", "1"]:
        os.makedirs(os.path.join(TARGET_ROOT, "all", label), exist_ok=True)

    patient_folders = [os.path.join(SOURCE_ROOT, d) for d in os.listdir(SOURCE_ROOT)
                       if os.path.isdir(os.path.join(SOURCE_ROOT, d))]

    for folder in patient_folders:
        for label in ["0", "1"]:
            image_folder = os.path.join(folder, label)
            if not os.path.exists(image_folder):
                continue
            images = glob(os.path.join(image_folder, "*.png"))
            for img_path in images:
                fname = os.path.basename(img_path)
                dst = os.path.join(TARGET_ROOT, "all", label, f"{os.path.basename(folder)}_{fname}")
                shutil.copy(img_path, dst)

def split_dataset():
    for label in ["0", "1"]:
        all_images = os.listdir(os.path.join(TARGET_ROOT, "all", label))
        random.shuffle(all_images)
        split_idx = int(len(all_images) * (1 - VAL_SPLIT))
        train_imgs = all_images[:split_idx]
        val_imgs = all_images[split_idx:]

        for split_type, images in zip(["train", "val"], [train_imgs, val_imgs]):
            split_folder = os.path.join(TARGET_ROOT, split_type, label)
            os.makedirs(split_folder, exist_ok=True)
            for img in images:
                src = os.path.join(TARGET_ROOT, "all", label, img)
                dst = os.path.join(split_folder, img)
                shutil.copy(src, dst)

def prepare_data_loaders():
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(TARGET_ROOT, "train"), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(TARGET_ROOT, "val"), transform=transform_val)

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    }
    sizes = {"train": len(train_ds), "val": len(val_ds)}
    return loaders, sizes

# ---------------------
# Model Variants
# ---------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

model_builders = {
    "simple_cnn": lambda: SimpleCNN(),
    "resnet50": lambda: (
        model := models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        setattr(model, 'fc', nn.Linear(model.fc.in_features, 1)),
        model
    )[-1],
    "vgg16": lambda: (
        model := models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        [p.requires_grad_(False) for p in model.features],
        setattr(model.classifier, '6', nn.Linear(4096, 1)),
        model
    )[-1]
}

loss_functions = {
    'BCE': nn.BCEWithLogitsLoss(),
    'MSE': nn.MSELoss()
}

optimizer_builders = {
    'Adam': lambda params: optim.Adam(params, lr=LR),
    'SGD': lambda params: optim.SGD(params, lr=LR, momentum=0.9)
}

scheduler_builders = {
    'None': None,
    'ReduceLROnPlateau': lambda opt: ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=3),
    'StepLR': lambda opt: StepLR(opt, step_size=7, gamma=0.1)
}

# ---------------------
# Training
# ---------------------
def train_model(model, loaders, sizes, criterion, optimizer, scheduler, epochs, patience=5):
    model.to(device)
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_acc": [], "val_acc": []}
    no_improve_epochs = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            corrects = 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = corrects.double() / sizes[phase]
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"No improvement for {no_improve_epochs} epoch(s)")

                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model)
    return model, history, best_acc

# ---------------------
# Main Experiment
# ---------------------
def main():
    if not os.path.exists(os.path.join(TARGET_ROOT, "train")):
        print("Preparing dataset...")
        collect_images()
        split_dataset()

    os.makedirs("saved_models", exist_ok=True)

    loaders, sizes = prepare_data_loaders()
    results = []

    for model_name, build_model in model_builders.items():
        for loss_name, loss_fn in loss_functions.items():
            for opt_name, opt_fn in optimizer_builders.items():
                model = build_model()
                optimizer = opt_fn(filter(lambda p: p.requires_grad, model.parameters()))
                for sched_name, sched_fn in scheduler_builders.items():
                    scheduler = sched_fn(optimizer) if sched_fn else None
                    tag = f"{model_name}_{loss_name}_{opt_name}_{sched_name}".replace("None", "none")
                    print(f"\n▶ Experiment: {tag}")
                    model, hist, best_acc = train_model(
                        model, loaders, sizes, loss_fn, optimizer, scheduler, EPOCHS, patience=PATIENCE
                    )

                    torch.save(model.state_dict(), f"saved_models/{tag}.pth")

                    results.append({
                        "experiment": tag,
                        "model": model_name,
                        "loss": loss_name,
                        "optimizer": opt_name,
                        "scheduler": sched_name,
                        "best_val_accuracy": best_acc.item()
                    })

    df = pd.DataFrame(results)
    df.to_csv("mammography_experiments.csv", index=False)
    print("\nTop Results:")
    print(df.sort_values(by="best_val_accuracy", ascending=False).head())

if __name__ == "__main__":
    main()
