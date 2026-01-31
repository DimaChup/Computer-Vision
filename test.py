import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINE DATA MANUALLY
# This is our specific case. No random generation.
X = np.array([
    [1, 2],  # Point 0
    [2, 1],  # Point 1
    [3, 1],  # Point 2
    [8, 8],  # Point 3
    [9, 8],  # Point 4
    [8, 9]   # Point 5
])

print(X)

c1 = np.array([1.0, 2.0]) 
c2 = np.array([5.0, 5.0])

plt.figure(figsize=(6, 6))
plt.scatter(X[:,0], X[:,1], s=100, label='Data')
plt.scatter(c1[0], c1[1], c='red', s=200, marker='X', label='Centroid 1 (Red)')
plt.scatter(c2[0], c2[1], c='blue', s=200, marker='X', label='Centroid 2 (Blue)')
plt.title("Step 1: The Setup")
plt.legend()
plt.grid(True)
# plt.show()

point = X[5] 

print("\n--- MANUAL MATH FOR POINT (8, 9) ---")

dist_to_red = np.sqrt(np.sum((point - c1)**2))
print(f"Distance to Red (1,2): {dist_to_red:.2f}")

dist_to_blue = np.sqrt(np.sum((point - c2)**2))
print(f"Distance to Blue (5,5): {dist_to_blue:.2f}")


if dist_to_red < dist_to_blue:
    print("-> Result: Point belongs to RED cluster")
else:
    print("-> Result: Point belongs to BLUE cluster")



red_group = []
blue_group = []

# We loop through every single point one by one
for i in range(len(X)):
    current_point = X[i]
    
    # Calculate distances
    d_red = np.sqrt(np.sum((current_point - c1)**2))
    d_blue = np.sqrt(np.sum((current_point - c2)**2))
    
    # Decide
    if d_red < d_blue:
        red_group.append(current_point)
    else:
        blue_group.append(current_point)

red_group = np.array(red_group)
blue_group = np.array(blue_group)

print(f"Points assigned to Red: \n{red_group}")
print(f"Points assigned to Blue: \n{blue_group}")

# Plot the groups
plt.figure(figsize=(6, 6))
plt.scatter(red_group[:,0], red_group[:,1], c='red', s=100, label='Red Team')
plt.scatter(blue_group[:,0], blue_group[:,1], c='blue', s=100, label='Blue Team')
plt.scatter(c1[0], c1[1], c='red', s=200, marker='X', edgecolors='black')
plt.scatter(c2[0], c2[1], c='blue', s=200, marker='X', edgecolors='black')
plt.title("Step 3: Teams Chosen")
plt.legend()
plt.grid(True)
#plt.show()

new_c1 = np.mean(red_group, axis=0)
new_c2 = np.mean(blue_group, axis=0)


# Plot the movement
plt.figure(figsize=(6, 6))
# Plot points
plt.scatter(red_group[:,0], red_group[:,1], c='red', s=100, alpha=0.3)
plt.scatter(blue_group[:,0], blue_group[:,1], c='blue', s=100, alpha=0.3)

# Plot OLD centroids (faint)
plt.scatter(c1[0], c1[1], c='red', s=150, marker='x', alpha=0.5, label='Old Red')
plt.scatter(c2[0], c2[1], c='blue', s=150, marker='x', alpha=0.5, label='Old Blue')

# Plot NEW centroids (solid)
plt.scatter(new_c1[0], new_c1[1], c='red', s=200, marker='X', label='New Red')
plt.scatter(new_c2[0], new_c2[1], c='blue', s=200, marker='X', label='New Blue')

# Draw arrows
plt.arrow(c1[0], c1[1], new_c1[0]-c1[0], new_c1[1]-c1[1], head_width=0.2, color='black')
plt.arrow(c2[0], c2[1], new_c2[0]-c2[0], new_c2[1]-c2[1], head_width=0.2, color='black')

plt.title("Step 4: Centroids Move to the Mean")
plt.legend()
plt.grid(True)
plt.show()