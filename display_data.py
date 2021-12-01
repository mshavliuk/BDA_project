import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

### Read csv file
df = pd.read_csv('heart.csv')

### Normalizing numeric features value
SS = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[col_to_scale] = SS.fit_transform(df[col_to_scale])
df.head()

# FINDING THE CORRELATION AMONG THE ATTRIBUTES
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)

sns.countplot(x='sex', data=df, hue='target') 
#### Female having heart disease are comparatively less when compared to males. 
#Males have low heart diseases as compared to females in the given dataset

sns.countplot(x='ca', hue='target', data=df)
# ca: number of major vessels (0-4) colored by flouroscopy

sns.countplot(x='thal', data=df, hue='target')
#thal 3 = normal, 6 = fixed defect, 7 = reversable defect

sns.countplot(x='cp', hue='target', data=df)
# cp (chest pain) 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
# Chest pain with respect to heart disease/target

sns.countplot(x='cp', hue='sex', data=df)
# chest pain count experienced by male and female

sns.barplot(x='sex', y='cp', hue='target', data=df, palette='cividis')

sns.barplot(x='sex', y='thal', data=df, hue='target')

sns.barplot(x='sex', y='ca', data=df, hue='target')
sns.barplot(x='sex', y='oldpeak', data=df, hue='target')
# depression induced by exercise relative to rest, a measure of abnormally in electrocardiograms

sns.barplot(x='fbs', y='chol', hue='target', data=df)
## Chorestorol level and target
