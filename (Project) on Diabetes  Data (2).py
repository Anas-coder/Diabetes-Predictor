#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Importing Data
data=pd.read_csv('diabetes2.csv')
data.head()


# In[40]:


data.dtypes


# # Understanding the data

# In[78]:


data.shape


# In[79]:


data.dtypes


# In[66]:


data.isna().sum()


# In[82]:


data.describe()


# In[83]:


from pandas_profiling import ProfileReport


# In[84]:


profile = ProfileReport(data,title='Pandas Profiling Report',explorative=True)


# In[85]:


profile.to_widgets()


# # Data Manipulation

# Checking The correlation of Data

# In[4]:


corrmat =data.corr() 
  
f, ax = plt.subplots(figsize =(15, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.9)


# In[6]:


#Correlation with output variable
cor_target = abs(corrmat["Outcome"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>=0.0001]
relevant_features.sort_values(ascending=False)


# Dropping The columns with correlation less then 0.10

# In[3]:


data.drop(['SkinThickness','BloodPressure'],axis=1,inplace=True)


# In[4]:


data


# In[5]:


X=data.iloc[:,0:6]

Y=data.Outcome


# In[7]:


Y


# Splitting The Data 

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


# # Model Building

# In[9]:


from sklearn.linear_model import LogisticRegression as lr


# In[10]:


clf = lr(random_state=42, solver='lbfgs',multi_class='multinomial')  #solver=lbfgh is gradient decent
clf.fit(X_train,y_train)


# In[11]:


clf.score(X_test,y_test)


# In[13]:


y_pred=clf.predict(X_test) 


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[54]:


from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
parameters = {  'max_leaf_nodes' : [2,3,4,5,6,7,8,9,10]}
grid = GridSearchCV(model,parameters, cv=None)
grid.fit(X_train, y_train)


# In[55]:


grid.best_params_


# In[16]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=150, min_samples_split=19, max_leaf_nodes=9, min_samples_leaf=3)
rnd_clf.fit(X_train,y_train) 


# In[22]:


y_pred=rnd_clf.predict(X_test)
y_pred.shape


# In[24]:


y_test.shape


# # Accuracy with k-Fold

# In[18]:


from sklearn.model_selection import cross_val_score
cross_val_score(rnd_clf,X_test,y_test ,cv=10, scoring="accuracy").mean()


# # Accuracy Check

# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[115]:


X.shape


# In[26]:


rnd_clf.predict([[6,98,190,34.0,0.430,43]])


# In[9]:


clf.predict([[1,189,60,23,846,30.1,0.398,59]])


# In[70]:


rnd_clf.score(X_test,y_test)


# In[73]:


get_ipython().set_next_input('from sklearn.neighbors import KNeighborsClassifier');get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')


# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=13)    #n_neighbors=13  root of total data and add 1
neigh.fit(X_train, y_train) 


# In[29]:


neigh.score(X_test,y_test)


# In[30]:


from sklearn.model_selection import cross_val_score
cross_val_score(neigh,X_train,y_train ,cv=10, scoring="accuracy").mean()


# In[31]:


import pickle


# In[40]:


# Saving model to disk
pickle.dump(rnd_clf, open('model.pkl','wb'))


# In[87]:


X


# In[41]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[ 4, 148,0,33.6,0.627,23]]))   #6	148	0	33.6	0.627	50

