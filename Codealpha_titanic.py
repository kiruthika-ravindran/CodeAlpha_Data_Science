#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


titanic_data=pd.read_csv("C:\\Users\\ADMIN\\Downloads\\titanic\\tested.csv")
titanic_data


# In[3]:


titanic_data.isnull()


# In[4]:


titanic_data.keys()


# In[5]:


titanic_data.head()


# In[6]:


titanic_data.tail()


# In[7]:


titanic_data.info()


# In[8]:


titanic_data.describe()


# In[9]:


titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['CabinKnown'] = titanic_data['Cabin'].notna().astype(int)

titanic_data.drop(columns=['Cabin'], inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)


# In[10]:


print(titanic_data.isnull().sum())
print(titanic_data.columns)
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)


# In[11]:


sns.boxplot(titanic_data['Fare'])


# In[12]:


plt.hist(titanic_data['Age'],bins=5)


# In[13]:


titanic_data['Survived'].value_counts().plot(kind='pie',autopct='%.3f')


# In[14]:


titanic_data['Pclass'].value_counts().plot(kind='pie',autopct='%.3f')


# ### DATA PREPROCESSING

# In[15]:


titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[16]:


print(titanic_data.isnull().sum())


# In[17]:


# Convert categorical variables to numerical
titanic_data_encoded = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)


# In[18]:


features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = titanic_data[features]
y = titanic_data['Survived']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


model = LogisticRegression(max_iter=200)  
model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[23]:


y_pred =logreg.predict(X_test)
y_pred


# In[24]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[25]:


conf_matrix = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix ,annot=True ,fmt='d',cmap='Blues',cbar= False)
plt.xlabel('predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




