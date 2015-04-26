print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import cPickle
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
# import some data to play with
iris = datasets.load_iris()
X=iris.data[:, [0,3]]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh
trainDataVecs = cPickle.load(open('save_train.p', 'rb'))
testDataVecs = cPickle.load(open('save_test.p', 'rb'))
X_train1, y_train = load_svmlight_file("data/train.txt")
X_test1, y_test = load_svmlight_file("data/test.txt")
X_train2, y_train2 = load_svmlight_file("data/train2.txt")
X_test2, y_test2 = load_svmlight_file("data/test2.txt")
X_train3=np.add(X_train1,X_train2)
X_test3=np.add(X_test1,X_test2)
X_train4=np.divide(X_train3,2)
X_test4=np.divide(X_test3,2)
newTrain = np.hstack((trainDataVecs, X_train4.toarray()))
newTest = np.hstack((testDataVecs, X_test4.toarray()))
X=newTrain
y=y_train
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
clf = svm.LinearSVC(C=C).fit(X, y)
print "Linear SVM"
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
print "RBF SVM"
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
print "Poly SVM"
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
print "LinearSVC"

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel']


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
#plt.subplot(1, 1, 1 + 1)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#print i
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z= clf.predict(newTest)

# Put the result into a color plot
#Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[i])

plt.show()
