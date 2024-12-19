import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def filter_classes(x, y, classes):
    """
    Filters the data by selecting only the specified classes.

    Args:
        x (np.ndarray): Input data (images).
        y (np.ndarray): Labels.
        classes (list): List of selected classes.

    Returns:
        np.ndarray, np.ndarray: Filtered data and labels.
    """
    idx = np.isin(y, classes).flatten()
    x_filtered = x[idx]
    y_filtered = y[idx]
    y_filtered = np.array([classes.index(label) for label in y_filtered.flatten()])
    return x_filtered, y_filtered


def normalize_data(x_train, x_test):
    """
    Normalizes the data by scaling pixel values to the range [0, 1].

    Args:
        x_train (np.ndarray): Training data.
        x_test (np.ndarray): Test data.

    Returns:
        np.ndarray, np.ndarray: Normalized training and test data.
    """
    return x_train / 255.0, x_test / 255.0


def plot_sample_images(x, y, class_names, num_images=6):
    """
    Displays sample images with titles.

    Args:
        x (np.ndarray): Input data (images).
        y (np.ndarray): Labels.
        class_names (list): Class names.
        num_images (int): Number of images to display.
    """
    plt.figure(figsize=(8, 4))
    for i in range(num_images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.show()


def plot_training_history(history):
    """
    Plots the training and validation accuracy.

    Args:
        history: The training history object (returned by model.fit).
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def display_confusion_matrix(y_true, y_pred, class_names):
    """
    Displays the confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (list): Class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.show()
