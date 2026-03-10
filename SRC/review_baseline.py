import pandas as pd
from collections import Counter

from paths import BASELINE_DIR

def main():
    df = pd.read_csv(BASELINE_DIR / "predictions.csv")

    print("Total predictions:", len(df))
    print()

    correct = (df["true_label"] == df["predicted_label"]).sum()
    incorrect = len(df) - correct

    print("Correct predictions:", correct)
    print("Incorrect predictions:", incorrect)
    print()

    print("Top predicted labels:")
    print(df["predicted_label"].value_counts())
    print()

    print("Top true labels:")
    print(df["true_label"].value_counts())
    print()

    mistakes = df[df["true_label"] != df["predicted_label"]]
    pair_counts = Counter(
        zip(mistakes["true_label"], mistakes["predicted_label"])
    )

    print("Most common mistakes:")
    for (true_label, predicted_label), count in pair_counts.most_common(15):
        print(f"{true_label} -> {predicted_label}: {count}")

if __name__ == "__main__":
    main()