import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# -----------------------
# Weather Data
# -----------------------
data = {
    "Date": pd.date_range("2025-01-01", periods=20),
    "Temperature": [25, 26, 27, 29, 31, 30, 28, 27, 26, 24, 23, 22, 21, 22, 23, 24, 25, 23, 21, 20],
    "Humidity": [60, 55, 62, 70, 68, 65, 60, 58, 55, 53, 58, 65, 68, 72, 70, 65, 62, 68, 75, 78],
    "Rainfall": [2, 0, 1, 5, 3, 1, 0, 0, 2, 4, 3, 5, 4, 6, 4, 2, 1, 3, 7, 8]
}

df = pd.DataFrame(data)

# -----------------------
# Create figure and subplots
# -----------------------
fig = plt.figure(figsize=(18, 12))
plt.suptitle("🌦️ Weather Dashboard (1D, 2D & 3D Visualizations)", fontsize=18, fontweight="bold")

# 1️⃣ Line Plot (Temperature)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df["Date"], df["Temperature"], color='red', marker='o')
ax1.set_title("Temperature Over Time (1D Line Plot)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (°C)")
ax1.grid(True)

# 2️⃣ Bar Plot (Rainfall)
ax2 = plt.subplot(3, 3, 2)
ax2.bar(df["Date"], df["Rainfall"], color='skyblue')
ax2.set_title("Rainfall per Day (2D Bar Plot)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Rainfall (mm)")
ax2.tick_params(axis='x', rotation=45)

# 3️⃣ Scatter Plot (Temperature vs Humidity)
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(df["Temperature"], df["Humidity"], color='green', s=80)
ax3.set_title("Temp vs Humidity (2D Scatter)")
ax3.set_xlabel("Temperature (°C)")
ax3.set_ylabel("Humidity (%)")
ax3.grid(True)

# 4️⃣ Histogram (Temperature Distribution)
ax4 = plt.subplot(3, 3, 4)
ax4.hist(df["Temperature"], bins=5, color='orange', edgecolor='black')
ax4.set_title("Temperature Distribution (1D Histogram)")
ax4.set_xlabel("Temperature (°C)")
ax4.set_ylabel("Frequency")

# 5️⃣ Boxplot (Rainfall)
ax5 = plt.subplot(3, 3, 5)
ax5.boxplot(df["Rainfall"], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
ax5.set_title("Rainfall Variation (Box Plot)")
ax5.set_ylabel("Rainfall (mm)")

# 6️⃣ Area Plot (Cumulative Rainfall)
ax6 = plt.subplot(3, 3, 6)
ax6.fill_between(df["Date"], df["Rainfall"].cumsum(), color='lightblue', alpha=0.6)
ax6.set_title("Cumulative Rainfall (2D Area Plot)")
ax6.set_xlabel("Date")
ax6.set_ylabel("Cumulative Rainfall (mm)")

# 7️⃣ 3D Scatter Plot
ax7 = fig.add_subplot(3, 3, 7, projection='3d')
ax7.scatter(df["Temperature"], df["Humidity"], df["Rainfall"], c=df["Rainfall"], cmap='viridis', s=80)
ax7.set_title("3D Weather Relationship")
ax7.set_xlabel("Temperature (°C)")
ax7.set_ylabel("Humidity (%)")
ax7.set_zlabel("Rainfall (mm)")

# 8️⃣ Heatmap (Correlation)
ax8 = plt.subplot(3, 3, 8)
corr = df[["Temperature", "Humidity", "Rainfall"]].corr()
im = ax8.imshow(corr, cmap='coolwarm', interpolation='nearest')
ax8.set_xticks(range(len(corr.columns)))
ax8.set_yticks(range(len(corr.columns)))
ax8.set_xticklabels(corr.columns)
ax8.set_yticklabels(corr.columns)
plt.colorbar(im, ax=ax8)
ax8.set_title("Correlation Heatmap")

# 9️⃣ Pie Chart (Rainfall Intensity Categories)
rainfall_categories = ["Low", "Medium", "High"]
counts = [sum(df["Rainfall"] <= 2), sum((df["Rainfall"] > 2) & (df["Rainfall"] <= 5)), sum(df["Rainfall"] > 5)]
ax9 = plt.subplot(3, 3, 9)
ax9.pie(counts, labels=rainfall_categories, autopct='%1.1f%%', startangle=90, colors=["#A2D2FF", "#FFB703", "#FB8500"])
ax9.set_title("Rainfall Intensity Distribution")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
