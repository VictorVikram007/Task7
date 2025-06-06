import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('/content/breast-cancer.csv')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(['id', 'diagnosis'], axis=1).values
y = df['diagnosis'].values
X_vis = X[:, :2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_vis_scaled = scaler.fit_transform(X_vis)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis_scaled, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print(classification_report(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_rbf))

params = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)

scores = cross_val_score(svm_rbf, X_scaled, y, cv=5)
print(scores)

svm_vis = SVC(kernel='linear', C=1)
svm_vis.fit(X_vis_train, y_vis_train)

plt.figure(figsize=(8, 6))
plot_decision_regions(X_vis_test, y_vis_test, clf=svm_vis, legend=2)
plt.title('Decision Boundary using Linear Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
