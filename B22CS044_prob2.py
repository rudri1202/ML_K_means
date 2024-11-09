# -*- coding: utf-8 -*-
"""B22CS044_prob2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dFi7njKA_VpFKbgppGZMYL68Y52VwKkf
"""

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons

iris = datasets.load_iris(as_frame=True)
iris

data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=['target'])

# Combine into a single DataFrame
df = pd.concat([data, target], axis=1)

# Print the DataFrame
print(df.head())

df = df[df['target'].isin([0, 1])]

#
X = df[['petal length (cm)', 'petal width (cm)']]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print the first few rows of the DataFrame
print(pd.DataFrame(X_scaled, columns=['petal length (cm)', 'petal width (cm)']).head())

clf = LinearSVC(random_state=42)
clf.fit(X_train, y_train)

# Plot the decision boundary on the training data
plt.figure(figsize=(10, 6))

# Create a meshgrid of feature values
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Make predictions on the meshgrid points
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot train data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Decision Boundary on Train Data')
plt.colorbar()

# Calculate accuracy on the train data
train_accuracy = accuracy_score(y_train, clf.predict(X_train))
plt.text(xx.max() - 0.3, yy.min() + 0.3, ('Train Accuracy: %.2f' % train_accuracy).lstrip('0'),
         size=12, horizontalalignment='right')

plt.show()

# Generate scatterplot of the test data along with the original decision boundary
plt.figure(figsize=(10, 6))

# Plot decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Test Data with Original Decision Boundary')
plt.colorbar()

# Calculate accuracy on the test data
test_accuracy = accuracy_score(y_test, clf.predict(X_test))
plt.text(xx.max() - 0.3, yy.min() + 0.3, ('Test Accuracy: %.2f' % test_accuracy).lstrip('0'),
         size=12, horizontalalignment='right')

plt.show()

X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

# Print the shape of the dataset
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Train SVM models with different kernels
svm_linear = SVC(kernel='linear', random_state=42)
svm_poly = SVC(kernel='poly', degree=3, gamma='auto', random_state=42)  # Polynomial kernel with degree=3
svm_rbf = SVC(kernel='rbf', gamma='auto', random_state=42)  # RBF kernel

# Fit the models
svm_linear.fit(X, y)
svm_poly.fit(X, y)
svm_rbf.fit(X, y)

# Plot decision boundaries for each kernel
plt.figure(figsize=(15, 5))

# Linear kernel
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Linear Kernel")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_linear.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(svm_linear.support_vectors_[:, 0], svm_linear.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

# Polynomial kernel
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Polynomial Kernel")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
Z = svm_poly.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

# Plot decision boundary
plt.contourf(XX, YY, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

# RBF kernel
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("RBF Kernel")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
Z = svm_rbf.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

# Plot decision boundary
plt.contourf(XX, YY, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10, 100]}

# Create the GridSearchCV object
grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5, n_jobs=-1)

# Perform grid search
grid_search.fit(X, y)

# Print the best parameters found
print("Best parameters:", grid_search.best_params_)

# Get the best SVM model
best_svm_rbf = grid_search.best_estimator_

# Train RBF kernel SVM model with the best hyperparameters
best_gamma = grid_search.best_params_['gamma']
best_C = grid_search.best_params_['C']
best_svm_rbf = SVC(kernel='rbf', gamma=best_gamma, C=best_C, random_state=42)
best_svm_rbf.fit(X, y)

# Plot decision boundary for the RBF kernel SVM with best hyperparameters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = best_svm_rbf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(best_svm_rbf.support_vectors_[:, 0], best_svm_rbf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for RBF Kernel SVM (Best Hyperparameters)')
plt.show()
