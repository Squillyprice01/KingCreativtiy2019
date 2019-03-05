import skflow
from sklearn import datasets, metrics
from sklearn.datasets import load_iris
data = load_iris()
iris = datasets.load_iris()
#feats = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
#try to find a minimal auto_examples
