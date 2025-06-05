# Task 7: Support Vector Machines (SVM) – AI & ML Internship

##  Objective
Use Support Vector Machines for linear and non-linear binary classification using the Breast Cancer Dataset. Apply data preprocessing, model training, decision boundary visualization, hyperparameter tuning, and performance evaluation.

##  Files Included
- `task7.py` – Complete SVM implementation in Python
- `breast-cancer.csv` – Dataset used for classification
- `task7-01.png`, `task7-02.png` – Screenshots of results/plots
- `README.md` – This file

##  Dataset Info
- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: Kaggle 
- **Target Classes**: Malignant (`M` = 1) and Benign (`B` = 0)
- **Features**: 30 numeric attributes after cleaning

##  Tools & Libraries
- Python 3
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Mlxtend (for 2D decision boundary visualization)

##  Steps Performed
1. Loaded and cleaned the dataset
2. Encoded the target variable (`M` → 1, `B` → 0)
3. Standardized features using `StandardScaler`
4. Trained SVM models with:
   - Linear kernel
   - RBF kernel
5. Visualized the decision boundary using the first 2 features
6. Tuned hyperparameters `C` and `gamma` using `GridSearchCV`
7. Evaluated model using:
   - Classification report
   - 5-fold cross-validation scores

##  Results
- **Linear Kernel Accuracy**: High precision & recall on both classes
- **RBF Kernel Accuracy**: Slightly better generalization
- **Best Params from Grid Search**: e.g., `C=1`, `gamma='scale'`
- Visual decision boundary plotted for 2D feature space (see screenshots)

##  What I Learned
- Importance of kernel selection in SVMs
- How SVM separates data using support vectors and margins
- Cross-validation and hyperparameter tuning for optimal performance
