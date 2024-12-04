import numpy as np
from matplotlib import pyplot as plt

def visualize_classifier(classifier, X, y, title='', mesh_step_size=0.1):
    """
    Visualizes the decision boundaries of a classifier in a 2D feature space.

    Parameters:
    - classifier: The trained classifier to visualize.
    - X: Array-like, shape (n_samples, 2). Input features (must be 2-dimensional for visualization).
    - y: Array-like, shape (n_samples,). Target labels for the samples.
    - title: String (optional). Title of the plot.
    - mesh_step_size: Float (optional). Step size for the mesh grid used to visualize decision boundaries.

    Steps:
    1. Define the range for the mesh grid based on the input feature range.
    2. Create a mesh grid of values with the specified step size.
    3. Use the classifier to predict outcomes for each point in the grid.
    4. Plot the decision boundaries as a colored background.
    5. Overlay the data points (features and their labels) on the plot.
    """
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure()
    plt.title(title)

    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    plt.yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    plt.show()