
# coding: utf-8

# In[3]:

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

path = '/Users/soledad/Box Sync/Fall 15/I590 - Collective Intelligence/CV Project/Code/svmethnicity/'
f = open(path+ '8svm.pkl', 'rb')
svm = pickle.load(f)
f.close()


train_set = np.load(path + '8train_set.pkl')
test_set = np.load(path + '8test_set.pkl')

labels_train=np.load(path + '8labels_train.pkl')
labels_test=np.load(path + '8labels_test.pkl')

predicted = svm.predict(test_set) 


# In[ ]:

names=['Happiness','Suprise', 'Sadness', 'Disgust', 'Fear', 'Anger']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(labels_test, predicted)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)


cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# In[5]:

labels_test


# In[ ]:




# In[6]:

plt.show()


# In[ ]:



