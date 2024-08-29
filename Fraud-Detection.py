import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv("Python//Credit-Card-Transaction//credit_card_transactions.csv")
df = data.dropna()

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'unix_time', 'merch_zipcode', 'trans_date_trans_time'], axis=1)

# Feature engineering: calculate distance
df['distance'] = ((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)**0.5

# Encode categorical variables
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)

# Separate features and target variable
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBClassifier
clf = XGBClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob)}")

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC Score: {pr_auc}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Find the best threshold
best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.1, 1.0, 0.01):
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_adjusted)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}")

# Evaluate model with best threshold
y_pred_best = (y_prob >= best_threshold).astype(int)
print("Classification Report with Best Threshold:")
print(classification_report(y_test, y_pred_best))
print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y_test, y_prob)}")
print(f"Precision-Recall AUC Score with Best Threshold: {pr_auc}")
print("Confusion Matrix with Best Threshold:")
print(confusion_matrix(y_test, y_pred_best))
