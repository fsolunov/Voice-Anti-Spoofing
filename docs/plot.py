import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path("logs/train.log")
CSV_PATH = Path("docs/history.csv")
FIG_PATH = Path("docs/training_curves.png")

PATTERN = re.compile(
    r"epoch\s+:\s+(\d+)\s+"
    r"loss\s+:\s+([\d.eE+-]+)\s+"
    r"grad_norm\s+:\s+([\d.eE+-]+)\s+"
    r"Accuracy\s+:\s+([\d.eE+-]+)\s+"
    r"dev_loss\s+:\s+([\d.eE+-]+)\s+"
    r"dev_Accuracy\s+:\s+([\d.eE+-]+)\s+"
    r"dev_EER\s+:\s+([\d.eE+-]+)"
)
FIELDS = ["epoch", "train_loss", "grad_norm", "train_accuracy",
          "dev_loss", "dev_accuracy", "dev_eer"]

text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
history = [
    dict(zip(FIELDS, [int(match[0])] + [float(value) for value in match[1:]]))
    for match in PATTERN.findall(text)
]
if not history:
    raise SystemExit(f"No epoch summaries found in {LOG_PATH}")

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(history)

epochs = [row["epoch"] for row in history]
best = min(history, key=lambda row: row["dev_eer"])

figure, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
axes[0].plot(epochs, [row["dev_loss"] for row in history], label="dev")
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("weighted cross-entropy")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, [row["dev_eer"] for row in history], color="tab:red")
axes[1].scatter([best["epoch"]], [best["dev_eer"]], color="black", zorder=5,
                label=f"selected checkpoint (epoch {best['epoch']}, "
                      f"{best['dev_eer']:.2f} %)")
axes[1].set_yscale("log")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("EER, %")
axes[1].set_title("Development EER")
axes[1].legend()
axes[1].grid(alpha=0.3)

figure.tight_layout()
figure.savefig(FIG_PATH, dpi=200)
print(f"{len(history)} epochs -> {CSV_PATH} and {FIG_PATH}")