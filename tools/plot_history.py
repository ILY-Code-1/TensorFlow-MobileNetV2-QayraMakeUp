"""
plot_history.py — Generate training_history.png dari training_log.txt.

Membaca training_log.txt, mengekstrak accuracy/val_accuracy/loss/val_loss
per epoch (kedua fase), lalu memplot dalam dua subplot:
  - Atas: Training & Validation Accuracy
  - Bawah: Training & Validation Loss

Output: training_history.png di root project.
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE = "training_log.txt"
OUT_FILE = "training_history.png"


def parse_log(path):
    """Ekstrak acc/val_acc/loss/val_loss per epoch dari log."""
    epochs = []
    phase1_end = None

    pattern = r"loss: ([\d.]+) - accuracy: ([\d.]+) - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)"

    with open(path, "r", encoding="utf-16-le", errors="replace") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                acc = float(match.group(2))
                val_loss = float(match.group(3))
                val_acc = float(match.group(4))
                epochs.append({
                    "loss": loss,
                    "acc": acc * 100,
                    "val_loss": val_loss,
                    "val_acc": val_acc * 100,
                })

    # Cari batas fase 1: cari baris "Epoch 1/10" setelah beberapa epoch
    # sebagai penanda masuk fase 2 (fine-tuning)
    with open(path, "r", encoding="utf-16-le", errors="replace") as f:
        content = f.read()
        # Cari pattern fase 2: "Epoch 1/10" (fase 2 pakai 10 epoch) atau "Epoch 1/20" setelah banyak epoch
        phase2_matches = list(re.finditer(r"Epoch 1/(\d+)\b", content))
        if len(phase2_matches) >= 2:
            # Fase 1 berakhir di posisi antara dua "Epoch 1/"
            idx = content[: phase2_matches[1].start()].count("- val_accuracy:")
            if idx > 0 and idx < len(epochs):
                phase1_end = idx

    return epochs, phase1_end


def plot_history(epochs, phase1_end):
    """Buat plot 2x1: accuracy (atas) dan loss (bawah)."""
    if not epochs:
        print("ERROR: tidak ada epoch yang berhasil diparse")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    x = list(range(1, len(epochs) + 1))
    train_acc = [e["acc"] for e in epochs]
    val_acc = [e["val_acc"] for e in epochs]
    train_loss = [e["loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]

    # ---- Accuracy ----
    ax1.plot(x, train_acc, "b-", label="Train Accuracy", linewidth=1.5)
    ax1.plot(x, val_acc, "r-", label="Validation Accuracy", linewidth=1.5)

    # Vertical line at phase transition
    if phase1_end:
        ax1.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="Fase 1 → Fase 2")
        ax2.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.6, label="Fase 1 → Fase 2")

    # Mark best val accuracy
    best_val_idx = val_acc.index(max(val_acc))
    ax1.plot(x[best_val_idx], val_acc[best_val_idx], "r*", markersize=12, label=f"Best Val: {val_acc[best_val_idx]:.2f}%")

    ax1.set_xlabel("Global Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training & Validation Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # ---- Loss ----
    ax2.plot(x, train_loss, "b-", label="Train Loss", linewidth=1.5)
    ax2.plot(x, val_loss, "r-", label="Validation Loss", linewidth=1.5)

    # Mark best val loss
    best_loss_idx = val_loss.index(min(val_loss))
    ax2.plot(x[best_loss_idx], val_loss[best_loss_idx], "r*", markersize=12, label=f"Best Val Loss: {val_loss[best_loss_idx]:.4f}")

    ax2.set_xlabel("Global Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    p1 = phase1_end if phase1_end else 0
    p2 = len(epochs) - p1
    print(f"training_history.png berhasil dibuat ({len(epochs)} epoch, {p1} Fase 1 + {p2} Fase 2)")
    print(f"Best val accuracy : {max(val_acc):.2f}% (epoch {best_val_idx + 1})")
    print(f"Best val loss     : {min(val_loss):.4f} (epoch {best_loss_idx + 1})")


if __name__ == "__main__":
    all_epochs, phase1_count = parse_log(LOG_FILE)
    if not all_epochs:
        print("ERROR: tidak ada epoch yang berhasil diparse dari training_log.txt")
    else:
        plot_history(all_epochs, phase1_count)
