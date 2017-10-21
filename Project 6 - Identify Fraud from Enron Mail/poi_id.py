#!/usr/bin/python
# ALEX MATHEW
import sys
import pickle
sys.path.append("../tools/")

import pprint
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from numpy import mean
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from tester import test_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#
features_list = ['poi'] # You will need to use more features
#WILL ADD MORE LATER !

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Exploring the data first !

num_data_points = len(data_dict)
num_data_features = len(data_dict[data_dict.keys()[0]])

num_poi = 0
for dic in data_dict.values():
	if dic['poi'] == 1: num_poi += 1

# For missing details
features_labels = data_dict['SKILLING JEFFREY K'].keys()
missing_stuff = {}
for feature in features_labels:
    missing_stuff[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in features_labels:
        if data_dict[person][feature] == 'NaN':
            missing_stuff[feature] += 1
        else:
            records += 1

print "The data points are : ", num_data_points
print "The features are : ", num_data_features
print "The POI's are : ", num_poi
print '\nMissing Data for Each Feature:\n'
for feature in features_labels:
    print feature,'\t',missing_stuff[feature],'\n'


# Removing outliers I found !

'''
Selecting salary and bonus as my features for finding.
Plotting them to view any outliers.
'''
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
import matplotlib.pyplot as plt
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("Salary Amount")
plt.ylabel("Bonus Amount")
print "\nOutliers checking !"
plt.show()

# From the pdf and mini-project code, we know the two outliers: 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL'
# For manual verification with pdf document
names = []
for i in data_dict:
    names.append(i)

names.sort()
pprint.pprint(names)


# 'THE TRAVEL AGENCY IN THE PARK' and 'TOTAL' are not individuals and hence are outliers and can be removed now!
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)

#Plotting again after removal of outliers !
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
print "\n After removal of outliers\n\n"
plt.show()

# Understanding whats left ! Creating outlier list
outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN': # Ignore
        continue
    outliers.append((key, int(value)))

extremeoutliers = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
print "The extreme most four outliers are :\n\n"
print extremeoutliers,'\n\n'
# Not removing these as they  are POI's

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


# I am adding two new features as follow !
# fraction_from_poi = Denotes the fraction of his/her emails this person recieved from POI individuals
# fraction_to_poi =   Denotes the fraction of his/her emails this person sent to POI individuals
for row in my_dataset:
	person = my_dataset[row]
	# Confirming we have all the data ! 
	if (person['from_poi_to_this_person'] != 'NaN' and
             person['from_this_person_to_poi'] != 'NaN' and
             person['to_messages'] != 'NaN' and
             person['from_messages'] != 'NaN'
             ):
		#calculating the two fractions
	    fraction_from_poi = float(person["from_poi_to_this_person"]) / float(person["from_messages"])
	    person["fraction_from_poi"] = fraction_from_poi
	    fraction_to_poi = float(person["from_this_person_to_poi"]) / float(person["to_messages"])
	    person["fraction_to_poi"] = fraction_to_poi
	else:
	    person["fraction_from_poi"] = person["fraction_to_poi"] = 0

# Creating my own feature list!
my_features = features_list + ['salary',
                                  'deferral_payments',
                                  'total_payments',
                                  'loan_advances',
                                  'bonus',
                                  'restricted_stock_deferred',
                                  'deferred_income',
                                  'total_stock_value',
                                  'expenses',
                                  'exercised_stock_options',
                                  'other',
                                  'long_term_incentive',
                                  'restricted_stock',
                                  'director_fees',
                                  'fraction_from_poi',
                                  'fraction_to_poi'
                                  ]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#displaying my features !
print "\n\nChosen features:\n\n", my_features


# Scaling the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# Finding the best feature to proceed with using K Best
# Going with 5 for k Value
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), my_features[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print "\n\nThe K-Best features are :\n", results_list,'\n\n'

#Choosing best 5 features

"""
Manually chose top 5 (since better results achieved !),
Tried with Top 4
Max possible is 5 as rest area false

#WITH TOP 4
my_features = features_list + ['salary',
                               'bonus',                               
                               'total_stock_value',
                               'exercised_stock_options',
                              ]
GaussianNB(priors=None)
        Accuracy: 0.84677       Precision: 0.50312      Recall: 0.32300 F1: 0.39342  
        F2: 0.34791    Total predictions: 13000        True positives:  646    False positives:  638  
        False negatives: 1354   True negatives: 10362


KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=3,           weights='distance')
        Accuracy: 0.84592       Precision: 0.49407      Recall: 0.06250 F1: 0.11096     F2: 0.07573
        Total predictions: 13000        True positives:  125    False positives:  128   
        False negatives: 1875   True negatives: 10872

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',    random_state=None, tol=0.001, verbose=0)
        Accuracy: 0.77254       Precision: 0.26715      Recall: 0.27450 F1: 0.27078   
        F2: 0.27300    Total predictions: 13000        True positives:  549    False positives: 1506  
        False negatives: 1451   True negatives: 9494


"""
my_features = features_list + ['salary',
                               'bonus',
                               'deferred_income',
                               'total_stock_value',
                               'exercised_stock_options',
                              ]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#For Gaussian Naieve Bayes
print '\n\nTrying Naieve Bayes Now:\n'
clf = GaussianNB()
print '\n\n'
test_classifier(clf, my_dataset, my_features)

#For  K neighbours
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_neighbors=10, p=3, weights='distance')
print '\n\nTrying K neighbours node Now:\n'
test_classifier(clf, my_dataset, my_features)

#For  K Means
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2, tol=0.001)
print '\n\nTrying KMeans Now:\n'
test_classifier(clf, my_dataset, my_features)

"""
#Calculated Values
GaussianNB(priors=None)
        Accuracy: 0.85464       Precision: 0.48876      Recall: 0.38050 F1: 0.42789     F2: 0.39814
        Total predictions: 14000        True positives:  761    False positives:  796 
        False negatives: 1239   True negatives: 11204

KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=10, p=3,           weights='distance')
        	
        Accuracy: 0.85793       Precision: 0.52174      Recall: 0.06600 F1: 0.11718     F2: 0.07997
        Total predictions: 14000        True positives:  132    False positives:  121
        False negatives: 1868   True negatives: 11879

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',    random_state=None, tol=0.001, verbose=0)
        Accuracy: 0.78729       Precision: 0.27192      Recall: 0.29150 F1: 0.28137     F2: 0.28736
        Total predictions: 14000        True positives:  583    False positives: 1561  
        False negatives: 1417   True negatives: 10439


"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
print 'Tuned KMeans\n\n'
from sklearn.cross_validation import train_test_split
# ACC TO SUGGESTION, ADDED STRATIFY ATTRIBUTE
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, stratify=labels,test_size=0.3, random_state=42)

# Tuning tol, n_init and max_iter
clf = KMeans(copy_x=True, init='k-means++', max_iter=500, n_clusters=2, n_init=20,
       n_jobs=1, precompute_distances='auto', random_state=None, tol=0.005,
      verbose=0)

test_classifier(clf, my_dataset, my_features)

clf = GaussianNB()
test_classifier(clf, my_dataset, my_features)
"""
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=500, n_clusters=2, n_init=20, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.005, verbose=0)
        Accuracy: 0.78471       Precision: 0.26996      Recall: 0.29750 F1: 0.28306     F2: 0.29155
        Total predictions: 14000        True positives:  595    False positives: 1609 
        False negatives: 1405   True negatives: 10391
 
 """
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features)