import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# 1️⃣ Generate Terrain Data
# -------------------------------
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Terrain height (Z) using sine & cosine waves
Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.3 * np.cos(X) * np.sin(Y)

# -------------------------------
# 2️⃣ Create Figure and Subplots
# -------------------------------
fig = plt.figure(figsize=(14, 6))

# ---- Plot 1: 3D Terrain Surface ----
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none')
ax1.set_title('3D Terrain Surface')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_zlabel('Height')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# ---- Plot 2: 2D Heatmap ----
ax2 = fig.add_subplot(1, 3, 2)
heat = ax2.imshow(Z, cmap='plasma', extent=[-5, 5, -5, 5], origin='lower')
ax2.set_title('2D Heatmap of Terrain')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
fig.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04)

# ---- Plot 3: 3D Contour Plot ----
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
contour = ax3.contour3D(X, Y, Z, 50, cmap='coolwarm')
ax3.set_title('3D Contour Plot')
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')
ax3.set_zlabel('Height')

# Adjust layout
plt.tight_layout()
plt.show()
