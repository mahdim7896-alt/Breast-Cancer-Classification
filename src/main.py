"""
Breast cancer classification project.

This project loads a dataset, processes the data, and trains a machine learning model
to predict whether a tumor is benign or malignant.
"""

#--- Import required libraries -------
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os 
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Paths Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
dataset_path = os.path.join(PROJECT_DIR, 'data', 'cancer (1).csv')


#--- Load the dataset and handle loading errors ---
try:
    df = pd.read_csv(dataset_path)
    #print('Success')
    print(df.head())
except:
    print('fail')


# Encode diagnosis labels (M → 1, B → 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis labels to numeric values


# Prepare features, normalize the data, 
# split into train/test sets, train the model, and make predictions
X=df.drop("diagnosis", axis=1)
X0=X.drop("id", axis=1)
X = (X0 - X0.min()) / (X0.max() - X0.min())   # Normalize features

y = df["diagnosis"]
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LogisticRegression(max_iter=10000)  # Train logistic regression model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#----- Plot and save the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")   # save image file
plt.show()