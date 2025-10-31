# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Load pandas, sklearn, and other required libraries for data handling, preprocessing, and SVM.
2. Load and Preprocess Data: Read the dataset with proper encoding, select relevant columns (text and label), and encode labels using LabelEncoder.
3. Split Dataset:Divide the data into training and testing sets using train_test_split.
4. Feature Extraction: Convert the email text into numerical features using TF-IDF Vectorization.
5. Train, Predict, and Evaluate: Train an SVM model with a linear kernel on the training data, predict labels for the test set, and evaluate using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sushmitha Gembunathan 
RegisterNumber:  212224040342
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
print(data.head())

print(data.shape)

x=data['v2'].values
y=data['v1'].values
print(x.shape)

print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

print(x_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("y_pred:\n",y_pred)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
print("accuracy:\n",acc)

con=confusion_matrix(y_test,y_pred)
print("confusion_matrix:\n",con)

cl=classification_report(y_test,y_pred)
print("classification:\n",cl)
```
## Output:
<img width="795" height="758" alt="image" src="https://github.com/user-attachments/assets/37106def-6eb5-4e45-adc1-f2d0e6db8a0f" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
