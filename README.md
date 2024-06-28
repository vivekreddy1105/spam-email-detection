# Email Spam Detection Using Logistic Regression

This project demonstrates how to build a simple email spam detection system using Logistic Regression. The goal is to classify email messages as either "spam" or "ham" (non-spam) based on their content. The dataset used for this project is a collection of email messages labeled as spam or ham.

## Libraries and Techniques Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms and tools, including:
  - `train_test_split`: To split the dataset into training and testing sets.
  - `TfidfVectorizer`: To convert text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
  - `LogisticRegression`: The machine learning model used for classification.
  - `accuracy_score`: To evaluate the accuracy of the model.

## Code Explanation

1. **Importing Libraries**:
   The necessary libraries are imported, including `pandas`, `numpy`, and several modules from `scikit-learn`.

2. **Loading and Preprocessing the Data**:
   The email dataset is loaded into a DataFrame using `pandas.read_csv()`. Missing values are handled by replacing them with empty strings. The labels ('Category' column) are converted to numerical values: 0 for spam and 1 for ham.

3. **Splitting the Data**:
   The data is split into training and testing sets using `train_test_split()`.

4. **Feature Extraction**:
   The text data is converted into numerical features using `TfidfVectorizer`, which transforms the email content into a TF-IDF matrix.

5. **Training the Model**:
   A Logistic Regression model is trained on the extracted features of the training data.

6. **Evaluating the Model**:
   The accuracy of the model is evaluated on the training data using `accuracy_score()`.

7. **Making Predictions**:
   A function is defined to predict whether new email messages are spam or ham using the trained model.

8. **Example Predictions**:
   The model is tested with a few example email messages to demonstrate its functionality.

