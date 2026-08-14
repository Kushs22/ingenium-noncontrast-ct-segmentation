"""
Generate 2x2 Overall Binary Confusion Matrix (All Vessels vs Background)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(r"C:\Users\zarin\Downloads\ai_healthcarehackathon\test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2x2 Matrix: [[TN, FP], [FN, TP]]
# Rows: True (0: Background, 1: Any Vessel)
# Cols: Pred (0: Background, 1: Any Vessel)
tn = 12987082
fp = 19345
fn = 14255
tp = 83287

cm_binary = np.array([
    [tn, fp],
    [fn, tp]
])

class_labels = ["Background", "Vessel (PA + Aorta)"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_binary, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=class_labels, yticklabels=class_labels,
       title="Overall Binary Confusion Matrix\n(All Vessels vs Background)",
       ylabel="True Ground Truth",
       xlabel="AI Prediction")

thresh = cm_binary.max() / 2.
descriptions = [
    [f"TN: {tn:,d}\n(Correctly Background)", f"FP: {fp:,d}\n(False Alarm)"],
    [f"FN: {fn:,d}\n(Missed Vessel)", f"TP: {tp:,d}\n(Detected Vessel)"]
]

for i in range(2):
    for j in range(2):
        ax.text(j, i, descriptions[i][j],
                ha="center", va="center",
                color="white" if cm_binary[i, j] > thresh else "black",
                fontsize=11, weight="bold")

plt.tight_layout()
binary_cm_path = OUTPUT_DIR / "confusion_matrix_overall_binary.png"
fig.savefig(binary_cm_path, dpi=200)
plt.close(fig)
print(f"Saved: {binary_cm_path}")
