import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.externals import joblib

# Read the dataset
df = pd.read_csv("https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/train_dataset.csv", index_col=0)

# Preprocessing
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with preprocessing, PCA, and an SVM classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=32)),
    ('classifier', SVC())
])

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__degree': [2, 3, 4],
    'classifier__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Save the best model to a file
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Print the best hyperparameters and cross-validated accuracy
print("Best hyperparameters: ", grid_search.best_params_)
print("Cross-validated accuracy: ", grid_search.best_score_)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("Accuracy on test set: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

