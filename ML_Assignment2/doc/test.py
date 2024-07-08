from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import Load_digits

digits = Load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(digits.data, digits.target, test_size=0.3)