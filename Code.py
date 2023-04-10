import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
data = pd.read_csv("corr_data.csv")

# Compute correlation matrix and p-values
corr_matrix, p_matrix = data.corr(), data.corr(method="pearson").apply(lambda x: np.round(x, 2))

# Add significance level information to p_matrix
# sig_matrix = p_matrix.applymap(lambda x: '*' if x <= 0.05 else 'NS')
sig_matrix = p_matrix.applymap(lambda x: '*' if x <= 0.05 else ('' if x == 1 else 'NS' if x > 1 else ''))

annot_matrix = p_matrix.astype(str) + sig_matrix

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=annot_matrix, fmt="", vmin=-1, vmax=1, cbar_kws={"label": "Correlation Coefficient"})

# Set plot title and axis labels
ax.set_title("Pearson Correlation Matrix")
ax.set_xlabel("Features")
ax.set_ylabel("Features")

# Show plot
plt.show()

# Save correlation matrix and p-values as CSV files
corr_matrix.to_csv("corr_matrix.csv")
p_matrix.to_csv("p_matrix.csv")
