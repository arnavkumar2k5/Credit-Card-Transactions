# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap

# Load and preprocess the data
# --------------------------------------
# Load the dataset
data = pd.read_csv("Python//Credit-Card-Transaction//credit_card_transactions.csv")
df = data.dropna()

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 
                   'zip', 'dob', 'trans_num', 'unix_time', 'merch_zipcode', 'trans_date_trans_time']
df = df.drop(columns=columns_to_drop, axis=1)

# Feature engineering: calculate distance between transaction location and merchant location
df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)

# Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Shuffle the dataset to ensure randomness
df = df.sample(frac=1, random_state=42)

# Split features and target variable
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model training and evaluation
# --------------------------------------
# Initialize and train the XGBoost classifier
clf = XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Evaluate the model using various metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc}")

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC Score: {pr_auc}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Optimize the decision threshold based on F1 score
# --------------------------------------
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 1.0, 0.01):
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}")

# Evaluate the model with the best threshold
y_pred_best = (y_prob >= best_threshold).astype(int)
print("Classification Report with Best Threshold:")
print(classification_report(y_test, y_pred_best))

roc_auc_best = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score with Best Threshold: {roc_auc_best}")
print(f"Precision-Recall AUC Score with Best Threshold: {pr_auc}")

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix with Best Threshold:")
print(conf_matrix_best)

# Feature importance visualization
# --------------------------------------
plot_importance(clf)
plt.show()

# Cross-validation for ROC AUC score
# --------------------------------------
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(f"Cross-Validated ROC AUC Scores: {cv_scores}")
print(f"Mean ROC AUC Score: {np.mean(cv_scores)}")

# SHAP analysis for model interpretability
# --------------------------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)
