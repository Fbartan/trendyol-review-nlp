import matplotlib.pyplot as plt

# Data
review_counts = [10000, 20000, 65000, 135000]
npmi_scores = [-0.1919, -0.2281, -0.0581, -0.0368]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(review_counts, npmi_scores, marker='o', linestyle='-', linewidth=2)
plt.title('NPMI Coherence Score vs. Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('NPMI Coherence Score')
plt.grid(True)
plt.xticks(review_counts)
plt.gca().invert_yaxis()  # Because lower negative values are worse
plt.tight_layout()
plt.show()
