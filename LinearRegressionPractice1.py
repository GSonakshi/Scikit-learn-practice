import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as datasets


iris = datasets.load_iris()
print("iris dataset is {}".format(iris.DESCR))
print("iris data size is {}".format(iris.data.shape))
print("iris target size is {}".format(iris.target.shape))
print("iris data has {} features, the feature names are {}".format(
    iris.data.shape[1], iris.feature_names))
print("iris data has {} samples, the target label names {}".format(
    iris.data.shape[1], iris.target_names))
