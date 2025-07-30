import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv("/content/adult 3.csv")

# Data Cleaning and Preprocessing
data['workclass'] = data['workclass'].replace({'?': 'Others'})
data['occupation'] = data['occupation'].replace({'?': 'Others'})
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
data = data.drop(columns=['education'])  # redundant feature

# Label Encoding (Fit on the entire data before splitting)
label_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
fitted_encoders = {} # Store fitted encoders
for col in label_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    fitted_encoders[col] = encoder # Store the fitted encoder

# Feature Scaling (for better convergence in some models like Logistic Regression, SVM)
scaler = StandardScaler()
features = data.drop(columns=['income'])
x = scaler.fit_transform(features)
y = data['income']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Selection and Training
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Gaussian Naive Bayes', GaussianNB())
]

trained_models = []
print("\nTraining models...")
for name, model in models:
    model.fit(x_train, y_train)
    trained_models.append((name, model))
    print(f"{name} training complete.")

# Model Evaluation
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
model_names = []

print("\nEvaluating models...")
for name, model in trained_models:
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    model_names.append(name)
    print(f"{name} evaluation complete.")

# Results DataFrame
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-score': f1_scores
})

# Display Results
print("\nModel Performance Results:")
print(results_df)

# Find and print the best model based on Accuracy
best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx]
print(f"\nBest model: {best_model['Model']} with accuracy {best_model['Accuracy']:.4f}")

# Predict employee salary using the best model (Gradient Boosting assumed best)

# Step 1: Prepare a sample employee input (must match scaled and encoded format)
# Example raw input (you would normally collect this from user or form):
sample_input = {
    'age': 35,
    'workclass': 'Private',
    'fnlwgt': 200000,
    'education': "B-tech", # This feature was dropped during preprocessing
    'educational-num': 13,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'gender': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 45,
    'native-country': 'United-States'
}
print()
print("Sample-Input :",sample_input)

# Step 2: Encode and scale the input (use same encoders/scalers from training)
sample_df = pd.DataFrame([sample_input])

# Drop the 'education' column as it was dropped from the training data
if 'education' in sample_df.columns:
    sample_df = sample_df.drop(columns=['education'])

# Apply label encoding to categorical columns using the fitted encoders
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    if col in sample_df.columns and col in fitted_encoders:
        try:
             sample_df[col] = fitted_encoders[col].transform(sample_df[col])
        except ValueError as e:
             print(f"Error encoding column {col}: {e}") # Handle unseen labels
             sample_df[col] = -1 # Assign a placeholder for showing of the error
    elif col in sample_df.columns and col not in fitted_encoders:
         print(f"Warning: Fitted encoder for feature '{col}' not found.")# Handle missing encoder, 
         sample_df[col] = -1 # Assign a placeholder
    elif col not in sample_df.columns and col in fitted_encoders:
         print(f"Warning: Feature '{col}' is missing from sample data but an encoder exists.")
         # Handle missing feature, e.g., add the column with a default value or mean
         sample_df[col] = -1 # Assign a placeholder

# Ensure column order matches training features
# Add missing columns if any, and ensure order
training_cols = features.columns.tolist() # Use the columns from the 'features' DataFrame before scaling
sample_df = sample_df.reindex(columns=training_cols, fill_value=0) # Fill missing numerical with 0, categorical should be encoded


# Scale the features
sample_scaled = scaler.transform(sample_df)


# Step 3: Use the best trained model to predict
best_model_name = best_model['Model']
predicted_salary = None
for name, model in trained_models:
    if name == best_model_name:
        predicted_salary = model.predict(sample_scaled)[0]
        break

# Step 4: Display the predicted income class
if predicted_salary is not None:
    # Need to reverse transform the predicted label if the income column was also encoded
    # Assuming income was encoded as 0 for <=50K and 1 for >50K, need to use the income encoder
    # We can use the fitted_encoders['income'] to reverse transform
    if 'income' in fitted_encoders:
        predicted_income_class = fitted_encoders['income'].inverse_transform([predicted_salary])[0]
        print("\nPredicted Income Class:", predicted_income_class)
    else:
        print("\nCould not reverse transform predicted salary. Income encoder not found.")
else:
    print("\nCould not make a prediction.")
