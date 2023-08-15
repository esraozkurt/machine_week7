#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor


# In[1]:


# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="blue",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="pink",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[ ]:


# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[ ]:


# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[ ]:


#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


# In[ ]:


# %% SVM

from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  


# In[ ]:


# %% test
print("print accuracy of naive bayes algo: ",classifier.score(x_test,y_test))


# In[ ]:


y_pred = classifier.predict(x_test)  
y_pred


# In[ ]:





# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:





# In[ ]:




