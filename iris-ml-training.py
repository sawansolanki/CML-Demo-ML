#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


from sklearn.datasets import load_iris
iris = load_iris()


# In[4]:


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

model = DecisionTreeClassifier()


# In[24]:


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# with open("metrics.txt", "w") as outfile:
#     outfile.write("Accuracy Score for Model: " + str(accuracy) + "\n")

print("Accuracy:", accuracy)


# In[23]:


import seaborn as sns
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='Blues').set(title='Confusion metrix',xlabel='predicted values', ylabel='actual values')

plt.savefig('confusion-metrics.png')


# In[15]:

#precision, recall, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)



accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Create a bar plot
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy, precision, recall, fscore]

# plt.bar(labels, scores)
# plt.ylabel('Score')
# plt.title('Model Performance')
# plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1

# # Show the plot
# plt.show()

result_dict = dict(zip(labels, scores))
print(result_dict)

with open("metrics.txt", "w") as outfile:
    outfile.write("Evaluation metrices for model: " + str(result_dict) + "\n")


# In[ ]:





# In[17]:


with open('iris-model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[18]:


with open('iris-model.pkl', 'rb') as file:
    iris_model = pickle.load(file)


# In[19]:


test_data = pd.DataFrame(
    np.array([
        [3.1, 1.5, 1.4, 2.2],  # Sample 1
        [6.2, 2.9, 4.3, 1.3],  # Sample 2
        [2.3, 2.8, 1.4, 2.0]   # Sample 3
    ]),
    columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)


# In[20]:


prdct = iris_model.predict(test_data)


# In[21]:


prdct


# In[ ]:





# In[ ]:




