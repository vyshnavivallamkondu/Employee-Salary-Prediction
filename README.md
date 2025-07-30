# Employee Salary Prediction using Machine Learning

This project focuses on predicting whether an employee's salary is greater than $50K per year based on demographic and work-related attributes using various machine learning classification models.

## 📌 Problem Statement
To predict the salary class of an employee (<=50K or >50K) based on features like age, education level, occupation, hours worked per week, and more.

## ⚙️ Technologies Used
- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn

## 📊 Machine Learning Models Implemented
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

## 🔄 Workflow
1. **Data Cleaning and Preprocessing**
   - Replacing missing values
   - Filtering irrelevant records
   - Label encoding categorical variables
   - Feature scaling

2. **Model Training and Evaluation**
   - Training all models on training data
   - Evaluating on test data (accuracy, precision, recall, F1-score)
   - Selecting the best-performing model

3. **Prediction**
   - Using the best model to predict salary class for new employee input

## 🏆 Best Performing Model
The model with the highest accuracy and F1-score is selected as the best model.

## 📁 Dataset
- The dataset used is the Adult Income dataset (UCI Repository format).
- Contains features such as age, workclass, education, marital-status, occupation, race, gender, hours-per-week, etc.

## 📌 How to Run
1. Clone this repository  
2. Install the required libraries
3. Run the main script
   ``` bash
    employee_salary_prediction.py
   ```
## 🔮 Future Improvements
- Use more advanced models like XGBoost or deep learning

- Build a web app using Streamlit or Flask for user input
## 👤 Author
Vyshnavi Vallmkondu - 
AI/Ml Enthuastic





