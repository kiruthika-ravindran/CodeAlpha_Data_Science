#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import norm
import math
import matplotlib.pyplot as plt


# In[17]:


data =pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\data\\New folder\\ab_data.csv")


# In[18]:


data.head()


# In[4]:


data.isnull()


# In[5]:


data.describe()


# In[6]:


data.keys()


# In[7]:


data.info()


# In[16]:


# Calculate conversion rates
control_conversions = data[(data['group'] == 'control') & (data['converted'] == 1)].shape[0]
control_total = data[data['group'] == 'control'].shape[0]
control_rate = control_conversions / control_total

treatment_conversions = data[(data['group'] == 'treatment') & (data['converted'] == 1)].shape[0]
treatment_total = data[data['group'] == 'treatment'].shape[0]
treatment_rate = treatment_conversions / treatment_total


# In[14]:


# Pooled conversion rate
pooled_conversion_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)

# Standard error
standard_error = math.sqrt(pooled_conversion_rate * (1 - pooled_conversion_rate) * (1 / control_total + 1 / treatment_total))

# Z-score
z_score = (treatment_rate - control_rate) / standard_error

# P-value (two-tailed test)
p_value = 2 * (1 - norm.cdf(abs(z_score)))


# In[15]:


print(f'Control Group Conversion Rate: {control_rate:.4f}')
print(f'Treatment Group Conversion Rate: {treatment_rate:.4f}')
print(f'Z-Score: {z_score:.4f}')
print(f'P-Value: {p_value:.4f}')

if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference between the control and treatment groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the control and treatment groups.")


# In[12]:


labels = ['Control', 'Treatment']
conversion_rates = [control_rate, treatment_rate]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
bars = ax.bar(x, conversion_rates, width, label='Conversion Rate')

ax.set_ylabel('Conversion Rate')
ax.set_title('Conversion Rate by Group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars:
    height = bar.get_height()
    ax.annotate('{}'.format(round(height, 4)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()


# In[20]:


df = pd.DataFrame(data)
conversions = df.groupby('group')['converted'].sum()
total_users = df['group'].value_counts()

labels = conversions.index
sizes = conversions.values
explode = (0, 0.1)  
colors = ['#ff9999','#66b3ff'] 
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Conversion Rate by Group')
plt.axis('equal')  
plt.show()


# In[ ]:




