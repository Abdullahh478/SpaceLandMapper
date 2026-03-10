import pandas as pd
from PIL import Image
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json

from config import IMAGE_SIZE, RANDOM_SEED
from paths import DATA_DIR, BASELINE_DIR


def load_split(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image).astype("float32") / 255.0
        X.append(image_array.flatten())
        y.append(row["label"])

    return np.array(X), np.array(y)


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    X_train, y_train = load_split(DATA_DIR / "train.csv")

    print("Loading validation data...")
    X_val, y_val = load_split(DATA_DIR / "val.csv")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=RANDOM_SEED)
    )

    print("Training baseline model...")
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    cm = confusion_matrix(y_val, y_pred, labels=sorted(set(y_val)))

    metrics = {
        "model": "baseline_logistic_regression",
        "image_size": IMAGE_SIZE,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }

    with open(BASELINE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    predictions_df = pd.DataFrame({
        "true_label": y_val,
        "predicted_label": y_pred
    })
    predictions_df.to_csv(BASELINE_DIR / "predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Baseline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(BASELINE_DIR / "confusion_matrix.png")
    plt.close()

    print("Saved:")
    print(BASELINE_DIR / "metrics.json")
    print(BASELINE_DIR / "predictions.csv")
    print(BASELINE_DIR / "confusion_matrix.png")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()