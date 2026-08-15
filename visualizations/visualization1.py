import matplotlib.pyplot as plt
import numpy as np

stages = ["US-only\nbaseline", "Global\nbaseline", "Global\ntuned (final)"]
auc = [0.80, 0.8228, 0.8226]
accuracy = [0.74, 0.74, 0.75]

x = np.arange(len(stages))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, auc, width, label="ROC-AUC", color="#aed7ff")
ax.bar(x + width/2, accuracy, width, label="Accuracy", color="#66aaff")

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Across Project Iterations")
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.legend()
ax.bar_label(ax.containers[0], fmt="%.3f")
ax.bar_label(ax.containers[1], fmt="%.3f")

plt.tight_layout()
plt.savefig("viz1_model_iterations.png", dpi=200)
plt.show()