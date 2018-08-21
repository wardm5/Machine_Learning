#!/usr/bin/python

# K Nearest Neighbor
from sklearn import neighbors
import matplotlib.pyplot as plt
#  K-Nearest Neighbor Example Code, thanks to SKLearn Examples -  https://bit.ly/2AWVSHO
def kthNearest(X, y, n):
    n_neighbors = n
    h = .02  # step size in the mesh
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Sets axis limits
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap='winter')

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("2-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))
    plt.show()
#  Confusion Matrix Example Code, thanks to SKLearn Examples -  https://bit.ly/2MdyDU9
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized K-Nearest confusion matrix")
    else:
        print('K-Nearest Confusion matrix, without normalization')

    print(cm)
    print()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#-------------------------------  Setup  -------------------------------#
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
nearestNeighbors = 40  # set nearest neighbors

#------------------------------- Step 1. -------------------------------#
# Load red wine data.
import pandas as p
import numpy as np
rawData = p.read_csv(dataset_url, ';')
y = np.array(rawData.quality)
X = np.array(rawData.drop('quality', axis=1))
y = np.where(y>5,1,0)  # reduces quality from 10 to 2 classes (1 for good, 0 for bad).

#------------------------------- Step 2. -------------------------------#
# Preprocessing
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#------------------------------- Step 3. -------------------------------#
# Transformation
from sklearn.decomposition import PCA
pca = PCA(3)  # project from 11 to 3 dimensions
projected3D = pca.fit_transform(X.data)
pca = PCA(2)  # project from 11 to 3 dimensions
projected2D = pca.fit_transform(X.data)
pca = PCA(2)
projectedTest = pca.fit_transform(X.data)

#------------------------------- Step 4. -------------------------------#
# Models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression"]
classifiers = [                                                                 # Model Number
    KNeighborsClassifier(nearestNeighbors, weights='uniform'),                  # 0
    SVC(kernel="linear", C=0.025),                                              # 1
    SVC(gamma=2, C=1),                                                          # 2
    DecisionTreeClassifier(max_depth=7),                                        # 3
    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1),       # 4
    MLPClassifier(alpha=2),                                                     # 4
    AdaBoostClassifier(),                                                       # 5
    GaussianNB(),                                                               # 6
    QuadraticDiscriminantAnalysis(),                                            # 7
    LogisticRegression()]                                                       # 8

kthNearest(projected2D, y, nearestNeighbors)  # Runs K-Nearest Neighbor Graphic method

#------------------------------- Step 5. -------------------------------#
# Cross Validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
cv = 5
for name, clf in zip(names, classifiers):                   # for each name and model
    scores = cross_val_score(clf, projected2D, y, cv=cv)    # run cross validation
    print(name)                                             # print name of model
    print('%.2f%%' % (np.average(scores)*100))              # print average of cross validation
print()
# Evalutation & Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
conf_matrix_name = ["P", "N"]
X_train, X_test, y_train, y_test = train_test_split(projectedTest, y, test_size=.8, random_state=5)
y_predicted = classifiers[0].fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=conf_matrix_name,
                      title='K-Nearest Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=conf_matrix_name, normalize=True,
                      title='Normalized K-Nearest confusion matrix')
plt.show()

#------------------------------- Step 6. -------------------------------#
# Results
# http://scikit-learn.org/0.15/modules/model_evaluation.html
# https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt

print("Accuracy for K-Nearest Neighbors \n%.2f%%"  % (metrics.accuracy_score(y_test, y_predicted)*100))
print("F1 Score for K-Nearest Neighbors \n%.2f%%"  % (metrics.f1_score(y_test, y_predicted)*100))
print("Precision for K-Nearest Neighbors \n%.2f%%"  % (metrics.precision_score(y_test, y_predicted)*100))
print("Recall for K-Nearest Neighbors \n%.2f%%"  % (metrics.recall_score(y_test, y_predicted)*100))

# ROC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='Accuracy = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#-------------------------------- Plots --------------------------------#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 3D Plot of PCA (n=3)
fig = plt.figure(1)
plt.title('Projected 3D Data')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projected3D[:, 0], projected3D[:, 1], projected3D[:,2], s=8, c=y, edgecolor='none',
           alpha=0.8, cmap=plt.cm.get_cmap('winter', 2))
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 2D Plot of PCA (n=2)
fig = plt.figure(2)
plt.title('Projected 2D Data')
plt.scatter(projected2D[:, 0], projected2D[:, 1], s=8, c=y, edgecolor='none',
           alpha=0.8, cmap=plt.cm.get_cmap('winter', 2))
plt.colorbar(ticks=[0,1])
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

#-------------------------- Correlation Plot ----------------------------#
corr = p.DataFrame(rawData)
correlations = corr.corr()
plt.matshow(correlations)
plt.title('Correlation Matrix')
ticks = np.arange(0,11,1)
plt.colorbar()
plt.show()