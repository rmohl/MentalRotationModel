import numpy as np

n = 8  # number of neurons
preferred_directions = np.linspace(0, 2 * np.pi, n, endpoint=False)  # create direction angles (in radians)

print(preferred_directions)

direction_vectors = np.array([
    [np.sin(theta), 0, np.cos(theta)]  # Y-component is zero
    for theta in preferred_directions
])

print(direction_vectors)
