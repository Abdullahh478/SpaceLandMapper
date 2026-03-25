import pandas as pd
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import IMAGE_SIZE, RANDOM_SEED, CLASS_NAMES
from paths import DATA_DIR, CNN_DIR


class EuroSATDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image).astype("float32") / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW

        label = self.label_to_idx[row["label"]]

        return torch.tensor(image_array, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    return np.array(all_true), np.array(all_preds)


def main():
    torch.manual_seed(RANDOM_SEED)
    CNN_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train_dataset = EuroSATDataset(DATA_DIR / "train.csv")

    print("Loading validation data...")
    val_dataset = EuroSATDataset(DATA_DIR / "val.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print("Training CNN model...")
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    print("Evaluating on validation set...")
    y_true_idx, y_pred_idx = evaluate(model, val_loader, device)

    y_true = [CLASS_NAMES[i] for i in y_true_idx]
    y_pred = [CLASS_NAMES[i] for i in y_pred_idx]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    metrics = {
        "model": "cnn",
        "image_size": IMAGE_SIZE,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }

    with open(CNN_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    predictions_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })
    predictions_df.to_csv(CNN_DIR / "predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CNN_DIR / "confusion_matrix.png")
    plt.close()

    print("Saved:")
    print(CNN_DIR / "metrics.json")
    print(CNN_DIR / "predictions.csv")
    print(CNN_DIR / "confusion_matrix.png")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()