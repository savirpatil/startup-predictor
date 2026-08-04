import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic\nRegression", "Random\nForest", "Neural\nNetwork"]
auc = [0.8226, 0.809, 0.81]        
accuracy = [0.75, 0.756, 0.74]    

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, auc, width, label="ROC-AUC", color="#aed7ff")
ax.bar(x + width/2, accuracy, width, label="Accuracy", color="#66aaff")

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Family Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.bar_label(ax.containers[0], fmt="%.3f")
ax.bar_label(ax.containers[1], fmt="%.3f")

plt.tight_layout()
plt.savefig("viz2_model_comparison.png", dpi=200)
plt.show()