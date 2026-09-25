import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "data/eye_features.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["timestamp"],
        df["average_ear"],
        linewidth=1,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Average EAR")
    plt.title("StudyVision - Eye Aspect Ratio During Paper Study")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()