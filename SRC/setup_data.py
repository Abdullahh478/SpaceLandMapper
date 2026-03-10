from pathlib import Path
import json

from config import CLASS_NAMES
from paths import DATA_DIR

def main():
    DATA_DIR.mkdir(exist_ok=True)

    label_map = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

    label_map_path = DATA_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    print("Created:", label_map_path)
    print("Class labels:")
    for name, idx in label_map.items():
        print(f"{idx}: {name}")

if __name__ == "__main__":
    main()