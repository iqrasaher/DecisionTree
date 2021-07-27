from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np

dataset = np.array([
['overcast','hot','high','FALSE','yes'],
['overcast','cool','normal','TRUE','yes'],
['overcast','mild','high','TRUE','yes'],
['overcast','hot','normal','FALSE','yes'],
['rainy','mild','high','FALSE','yes'],
['rainy','cool','normal','FALSE','yes'],
['rainy','cool','normal','TRUE','no'],
['rainy','mild','normal','FALSE','yes'],
['rainy','mild','high','TRUE','no'],
['sunny','hot','high','FALSE','no'],
['sunny','hot','high','TRUE','no'],
['sunny','mild','high','FALSE','no'],
['sunny','cool','normal','FALSE','yes'],
['sunny','mild','normal','TRUE','yes']
])
#print(dataset)
features =dataset[:,0:-1]
#print(features)
target = dataset[:,-1]
#print(target)
enc=preprocessing.OrdinalEncoder()
enc.fit(features)
Tfeatures = enc.transform(features)
#print(Tfeatures)
Dtree =DecisionTreeClassifier(criterion='entropy')
fitted =Dtree.fit(Tfeatures,target)
Guess = np.array([["sunny","mild","normal","FALSE"]])
prediction=fitted.predict(enc.transform(Guess))
print(prediction)
