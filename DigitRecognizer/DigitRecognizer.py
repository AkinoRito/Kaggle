import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


def normalizing(array):
    array = np.mat(array)
    m, n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j] > 0:
                array[i,j] = 1
    return array


labeled_images = pd.read_csv("train.csv")
images = labeled_images.iloc[:, 1:]
labels = labeled_images.iloc[:, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
train_images = normalizing(train_images)
test_images = normalizing(test_images)

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print("training acc: ", clf.score(test_images, test_labels))

test_data = pd.read_csv("test.csv")
test_data = normalizing(test_data)
result = clf.predict(test_data)

df = pd.DataFrame(result)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('result.csv', header=True)
