
# coding: utf-8

# In[5]:


#Random forest vs Decision Tree classification report and analysis on load_data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('loan_data.csv')


# In[7]:


df.head()


# In[10]:


df.describe()


# In[31]:


plt.figure(figsize=(10,7))
df[df['credit.policy']==1]['fico'].hist(color='blue',
                                              bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
#fico -  credit score of the borrower.


# In[32]:


df.head()


# In[45]:


plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,bins=30,color='red',label='fully paid')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,bins=30,color='blue',label='not fully paid')
plt.legend()
plt.xlabel('credit score of the borrower.')


# In[47]:


#check if purpose has any kind of relation with payment
plt.figure(figsize=(11,7))
sb.countplot(x='purpose',hue='not.fully.paid',data=df)
#df.columns


# In[49]:


#credit score and interest rate relational behavior 
sb.jointplot(x='fico',y='int.rate',data=df)


# In[52]:


plt.figure(figsize=(11,7))
sb.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',col='not.fully.paid')


# In[53]:


#setting up the data 
df.info()


# In[54]:


df.head()


# In[55]:


categorical_feat =['purpose']


# In[56]:


final_df = pd.get_dummies(df,columns=categorical_feat,drop_first=True)


# In[57]:


final_df.head()


# In[61]:


#New spliting the data into test and train set 
from sklearn.model_selection import train_test_split


# In[63]:


X = final_df.drop('not.fully.paid',axis=1)
y = final_df['not.fully.paid'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[65]:


# Training decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[66]:


dtree.fit(X_train,y_train)


# In[67]:


#Get the predictions from the test set and create a classification report and a confusion matrix.
predictions = dtree.predict(X_test)


# In[69]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# In[70]:


print(confusion_matrix(y_test,predictions))


# In[72]:


#Training the model using Random Forest algorithm


# In[73]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


rfc = RandomForestClassifier(n_estimators=350)


# In[86]:


rfc.fit(X_train,y_train)


# In[87]:


#now prediction using test data
predict = rfc.predict(X_test)


# In[88]:


# now creating classification report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))


# In[89]:


print(confusion_matrix(y_test,predict))

