#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, auc


# In[2]:


import pickle
import numpy as np


# In[3]:


# Load the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()


# In[4]:


# Create a DataFrame from the Iris dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']


# In[5]:


df.shape


# In[6]:


df = df.drop(['petal length (cm)'], axis=1)


# In[7]:


df.head(5)


# In[8]:


y.unique()


# In[9]:


df.corr()


# In[10]:


y.shape


# In[11]:


X.shape


# In[12]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# Create a Random Forest Classifier model
model = DecisionTreeClassifier()


# In[13]:


# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy Score for Model: " + str(accuracy) + "\n")

print("Accuracy:", accuracy)


# In[14]:


import seaborn as sns
from sklearn.metrics import confusion_matrix


# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn
sns.heatmap(cm, annot=True, cmap='Blues').set(title='Confusion metrix',xlabel='predicted values', ylabel='actual values')


# In[100]:


# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Create a bar plot
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy, precision, recall, fscore]

plt.bar(labels, scores)
plt.ylabel('Score')
plt.title('Model Performance')
plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1

# Show the plot
plt.show()


# In[113]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)

# Plot the Precision-Recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Show the plot
plt.savefig('Precision-Recall-Curve.png')
plt.show()


# In[103]:


# Save the model to a .pkl file
with open('iris-model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[32]:


with open('iris-model.pkl', 'rb') as file:
    iris_model = pickle.load(file)


# In[44]:


test_data = pd.DataFrame(
    np.array([
        [3.1, 1.5, 1.4, 2.2],  # Sample 1
        [6.2, 2.9, 4.3, 1.3],  # Sample 2
        [2.3, 2.8, 1.4, 2.0]   # Sample 3
    ]),
    columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)


# In[45]:


prdct = iris_model.predict(test_data)


# In[46]:


prdct


# In[ ]:
