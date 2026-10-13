import matplotlib.pyplot as plt
import os
from pathlib import Path

def parse_log(file_path):
    epochs = []
    losses = []
    accuracies = []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            try:
                parts = line.strip().split("|")

                epoch = int(parts[0].split()[1])
                loss = float(parts[1].split(":")[1])
                acc = float(parts[2].split(":")[1].replace("%", ""))

                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc)

            except Exception as e:
                print(f"Skipping line: {line.strip()}")

    return epochs, losses, accuracies


def plot_data(epochs, losses, accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss plot (left axis)
    ax1.plot(epochs, losses, marker='o', linestyle='-', label='Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Log scale (optional but useful for large loss values)
    ax1.set_yscale('log')

    # Accuracy plot (right axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, marker='s', linestyle='--', label='Accuracy')
    ax2.set_ylabel("Accuracy (%)")

    # Title & grid
    plt.title("Model Training Performance")
    ax1.grid(True)

    # Combined legend
    fig.legend(loc="upper right")

    # Save image
    plt.savefig("training_graph.png")

    # Show graph
    plt.show()


if __name__ == "__main__":
    file_path = Path(os.environ.get("TRAINING_LOG_PATH", Path(__file__).resolve().parent / "saved_models_93" / "modelaccuracy.log"))

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Training log not found at {file_path}. Place it there or set TRAINING_LOG_PATH."
        )

    epochs, losses, accuracies = parse_log(file_path)

    if len(epochs) == 0:
        print("No valid data found!")
    else:
        plot_data(epochs, losses, accuracies)
