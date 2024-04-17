import seaborn as sns
import matplotlib.pyplot as plt

# Define the methods and their accuracies
methods = ['Label-Conditional CP', 'Random Forest', 'Transductive CP', 'Inductive CP']
accuracies = [0.8831168831168831, 0.7207792207792207, 0.8831168831168831, 0.896551724137931]

# Create a Seaborn bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=methods, y=accuracies, palette="husl")

# Add labels on top of each bar
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

# Set plot labels and title

plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to ensure all values are displayed
plt.show()
