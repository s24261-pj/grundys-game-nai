import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from meet_4.decision_tree.utilities import visualize_classifier

"""
Load the dataset and preprocess it. 
The dataset is assumed to be in CSV format, with features in columns except for the last column, which contains labels.
"""
input_file = './../files/pima_diabetes.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

"""
Standardize features by removing the mean and scaling to unit variance. 
This step is important for PCA to ensure that all features contribute equally to the dimensionality reduction.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Perform dimensionality reduction using PCA (Principal Component Analysis). 
Reduce the number of features to 2 for easier visualization while retaining as much variance as possible.
"""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

"""
Split the dataset into training and testing subsets.
Use a 75% training and 25% testing split, and set a random state for reproducibility.
"""
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.25, random_state=5)

"""
Train a Decision Tree Classifier on the training dataset.
Specify hyperparameters such as `random_state` for reproducibility and `max_depth` to prevent overfitting.
"""
params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train_pca, y_train_pca)

"""
Evaluate the classifier's performance using the classification report, 
which provides precision, recall, and F1-score for each class on both training and testing datasets.
"""
class_names = ['Class-0', 'Class-1']
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train_pca, classifier.predict(X_train_pca), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test_pca, classifier.predict(X_test_pca), target_names=class_names))
print("#" * 40 + "\n")

"""
Visualize the decision boundaries of the trained classifier using a custom utility function.
This visualization is done for both training and testing datasets with a specified grid resolution (`mesh_step_size`).
"""
visualize_classifier(classifier, X_train_pca, y_train_pca, title="Pima Indians Diabetes decision Tree Train Classifier (PCA)", mesh_step_size=0.1)

visualize_classifier(classifier, X_test_pca, y_test_pca, title="Pima Indians Diabetes decision Tree Test Classifier (PCA)", mesh_step_size=0.1)