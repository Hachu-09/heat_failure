# train_heart_failure_model.py

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
heart_failure = pd.read_csv("HEART_MODEL/Heart Failure/heart_failure.csv")

# Separate features and target variable
x = heart_failure.drop(columns="HeartDisease", axis=1)
y = heart_failure["HeartDisease"]

# Scale features for better performance of some models
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Initialize individual models with adjusted parameters
log_reg = LogisticRegression(max_iter=200)
svm = SVC(kernel='linear', probability=True)
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
gb = GradientBoostingClassifier()

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('svm', svm),
        ('rf', rf),
        ('knn', knn),
        ('gb', gb)
    ],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(x_train, y_train)

# Save the trained model to a pickle file
with open('HEART_MODEL/Heart Failure/heart_failure_prediction.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)

# Save the scaler to a pickle file
with open('HEART_MODEL/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler trained and saved successfully.")
