from pathlib import Path
from config import CLASS_NAMES
from paths import DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    print("Scanning dataset...\n")

    total_images = 0

    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            print(f"[MISSING] {class_name}")
            continue

        image_count = sum(
            1 for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += image_count
        print(f"{class_name}: {image_count} images")

    print(f"\nTotal images: {total_images}")


if __name__ == "__main__":
    main()