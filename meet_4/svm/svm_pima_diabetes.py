from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

"""
Load and preprocess the dataset for classification.
- The dataset is expected to be a CSV file with features and labels in a specific format.
- The features and labels are encoded using LabelEncoder to transform categorical data into numerical values.
"""
input_file = './../files/pima_diabetes.csv'
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

"""
Standardize the feature set using StandardScaler.
- Standardization ensures each feature has mean 0 and variance 1, which is important for PCA and SVM.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
Perform Principal Component Analysis (PCA) to reduce the dimensionality of the feature set to 2 components.
- This allows for easier visualization while retaining the most important information.
"""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

"""
Split the dataset into training and testing sets.
- 75% of the data is used for training, and 25% is used for testing.
- A random state is set for reproducibility of the split.
"""
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=5)

"""
Train a Support Vector Classifier (SVC) with an RBF kernel.
- The SVC is trained on the training set with specified hyperparameters (C, gamma, random_state).
"""
svc = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
svc.fit(X_train, y_train)

"""
Make predictions on the training and testing datasets.
- The predictions are compared to the true labels to evaluate the model's performance.
"""
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)

"""
Evaluate the model using classification reports.
- This provides precision, recall, and F1-score for both training and test datasets.
"""
class_names = ['Class-0', 'Class-1']
print("\n" + "#" * 40)
print("\nSVM performance on training dataset\n")
print(classification_report(y_train, y_train_pred, target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nSVM performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

"""
Visualize the decision boundary of the trained SVM classifier in 2D using the first two PCA components.
- The plot shows how the classifier separates the data into two classes, with a contour plot representing the decision boundary.
"""
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Class 0', c='blue', edgecolors='k')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Class 1', c='red', edgecolors='k')

plt.title('SVM Decision Boundary with PCA (Normalized)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.show()