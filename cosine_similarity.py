import matplotlib.pyplot as plt
import numpy as np

# Define three vectors
vec1 = np.array([1, 2])
vec2 = np.array([2, 1])
vec3 = np.array([-1, -1])  # A third vector, less similar to vec1

# Normalize vectors
vec1_norm = vec1 / np.linalg.norm(vec1)
vec2_norm = vec2 / np.linalg.norm(vec2)
vec3_norm = vec3 / np.linalg.norm(vec3)

# Cosine similarities
cos_similarity_12 = np.dot(vec1_norm, vec2_norm)
cos_similarity_13 = np.dot(vec1_norm, vec3_norm)

# Create a 4th vector orthogonal to vec1
orthogonal_vec = np.array([-vec1[1], vec1[0]])  # Rotate vec1 by 90 degrees
orthogonal_vec_norm = orthogonal_vec / np.linalg.norm(orthogonal_vec)
cos_similarity_14 = np.dot(vec1_norm, orthogonal_vec_norm)  # Should be ~0

# Plotting the vectors
fig, ax = plt.subplots()
ax.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
ax.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=1, color='red', label='Vector 2')
ax.quiver(0, 0, vec3[0], vec3[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector 3')
ax.quiver(0, 0, orthogonal_vec[0], orthogonal_vec[1], angles='xy', scale_units='xy', scale=1, color='purple', label='Orthogonal Vector')

# Setting up the plot
plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title(f"Cosine Similarity Visualization\n"
          f"Vec1-Vec2: {cos_similarity_12:.2f}, Vec1-Vec3: {cos_similarity_13:.2f}, Vec1-Orthogonal: {cos_similarity_14:.2f}")
plt.show()