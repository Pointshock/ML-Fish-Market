# Import libraries
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# We will need to use text based data for our decision tree
# so we will import the one hot encoder from sklearn
from sklearn.preprocessing import OneHotEncoder
# KNN Library
from sklearn.neighbors import KNeighborsClassifier
# ignore warnings from imports
import warnings
warnings.filterwarnings("ignore")

# Define a data file to import
fish_data_file = "Fish.csv"
# Import the csv data file into a pandas data frame
df = pd.read_csv(fish_data_file)

# Now it is time for feature selection
features = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]

# Now it is time for label/ target selection
label = df[['Species']]

# Split data into training and testing
# We will use the train_test_split function from sklearn
# We will use 90% of the data for training and 10% for testing
features_training_data, features_testing_data, label_training_data, label_testing_data = train_test_split(features, label, test_size=0.20, random_state=32)

# Let's prepare to use a decision tree classifier
# We will create and use the tree.DecisionTreeClassifier function from sklearn
classifier_dt = tree.DecisionTreeClassifier()
# KNN
knn_clf = KNeighborsClassifier()

# Now we will train the classifier - this is the machine learning part
# We will use the fit function from sklearn
classifier_dt = classifier_dt.fit(features_training_data, label_training_data)
# KNN
knn_clf = knn_clf.fit(features, label)

# Now we will test the classifier
# We will use the predict function from sklearn
label_predicted_data = classifier_dt.predict(features_testing_data)
# KNN
knn_clf_prediction = knn_clf.predict(features_testing_data)
print("\nThe KNN prediction is: ", knn_clf_prediction)
# Now we will use the accuract_score function from sklearn
# to see how accurate our classifier is
# Round the accuracy score for better UX
rounded_score_dt = round(accuracy_score(label_testing_data, label_predicted_data), 2)
# Turn into %
percent_score_dt = rounded_score_dt * 100
# Print out the accuracy score %
print("\nThis shows the accuracy score of a machine learning program based on the Fish Market data set from Kaggle.")
print("\n\n\tAccuracy Score: ", percent_score_dt,"%")
print("\n")
