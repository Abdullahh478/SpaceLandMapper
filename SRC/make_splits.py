from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, RANDOM_SEED, TEST_SIZE, VAL_SIZE
from paths import DATA_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_rows():
    rows = []

    for class_name in CLASS_NAMES:
        class_dir = DATA_DIR / class_name

        for file in class_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "image_path": str(file.resolve()),
                    "label": class_name
                })

    return pd.DataFrame(rows)


def main():
    df = collect_rows()

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_relative = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"],
    )

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("Created:")
    print(DATA_DIR / "train.csv", len(train_df))
    print(DATA_DIR / "val.csv", len(val_df))
    print(DATA_DIR / "test.csv", len(test_df))


if __name__ == "__main__":
    main()