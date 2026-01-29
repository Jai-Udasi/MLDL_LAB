import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/STUDENT/Desktop/MLDL 64/student_performance.csv")
df.head()
final_score_np = df["Final_Score"].values
print(final_score_np)
mean_score = np.mean(final_score_np)
median_score = np.median(final_score_np)
std_dev_score = np.std(final_score_np)

print("Mean:", mean_score)
print("Median:", median_score)
print("Standard Deviation:", std_dev_score)
min_score = np.min(final_score_np)
max_score = np.max(final_score_np)

normalized_scores = (final_score_np - min_score) / (max_score - min_score)

print("Minimum Final Score:", min_score)
print("Maximum Final Score:", max_score)
print("Normalized Final Scores:")
print(normalized_scores)
df = pd.read_csv("C:/Users/STUDENT/Desktop/MLDL 64/student_performance.csv")
print("Shape of dataset:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())
def performance_label(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Final_Score"].apply(performance_label)
df.head()
plt.figure()
plt.plot(df["Hours_Studied"], df["Final_Score"], marker='o')
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.title("Hours Studied vs Final Score")
plt.show()
plt.figure()
plt.hist(df["Final_Score"], bins=10)
plt.xlabel("Final Score")
plt.ylabel("Frequency")
plt.title("Distribution of Final Scores")
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.scatterplot(
    x="Hours_Studied",
    y="Final_Score",
    hue="Performance",     # color by category
    palette="Set2",
    data=df,
    s=100                  # point size
)

plt.title("Colorful Scatter Plot: Hours Studied vs Final Score")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.show()

plt.figure()
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
plt.figure()
sns.boxplot(x="Performance", y="Final_Score", data=df)
plt.title("Final Score vs Performance Category")
plt.show()