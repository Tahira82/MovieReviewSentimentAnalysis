import pandas as pd #manipulates data-sets
import imblearn as imb
import sklearn as sk #to use machine learning models
import sklearn.feature_extraction.text as f_e
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#reading data from csv file
oData = pd.read_csv('IMDB Dataset.csv')

#reading imbalanced data from csv file: 9000 positive and 1000 negative
# sentiments
oImbalancedPosData = oData[oData['sentiment'] == 'positive'][:9000]
oImbalancedNegData = oData[oData['sentiment'] == 'negative'][:1000]
oImbalancedData = pd.concat([oImbalancedPosData, oImbalancedNegData]) #axis 0 by default: row wise concat

#using undersampling technique to resample imbalanced data
oRandomSampler = imb.under_sampling.RandomUnderSampler(random_state = 0)
#applying random undersampling on imbalanced data
#output: review(2000*1) and sentiment column attached so 2000*2
#fit sample arguments: matrix 10000*1, object labels
oDataBalanced, oDataBalanced['sentiment'] = oRandomSampler.fit_resample(oImbalancedData[['review']],
                                                                        oImbalancedData['sentiment'])
#no data cleaning required
#splitting test train data
matTrainData, matTestData = sk.model_selection.train_test_split(oDataBalanced, test_size = 0.33,
                                                                random_state= 42)

#labelling dependent y and independent x variables for data
oTrain_x, oTrain_y = matTrainData['review'], matTrainData['sentiment']
oTest_x, oTest_y = matTestData['review'], matTestData['sentiment']

#conversion of text data to numeric data as machines understand numbers
#using Bag of words technique as the word count is important for us not the order.
#Bag of words cannot take into account the sequence of words and it is not our requirement either
#Implementing Bag of words using TD-IDF and not Count Vectorizer technique because count vectorizer
#only takes into account the frequency of words while TF-IDF also points out the unique words for classification
#fit and transform(apply params) test and training x data to vector
oTfidfModel = f_e.TfidfVectorizer(stop_words='english')
matTrain_x = oTfidfModel.fit_transform(oTrain_x) #1340*20625
matTest_x = oTfidfModel.transform(oTest_x) #already fit(find parameters) tfid model

#selecting model:
#1. supervised: labelled data
#2. classification algorithms: output is either positive or negative
#3. finding scores for svm(support vector machines), logistic regression, naive bayes, decision tree

#models initialization
oSVC = SVC(kernel='linear') #support vector machine
oDTC = DecisionTreeClassifier() #decision tree
oNB = GaussianNB() #naive bayes
oLR = LogisticRegression() #logistic regression

#fit model parameters
oSVC.fit(matTrain_x, oTrain_y)
oDTC.fit(matTrain_x, oTrain_y)
oNB.fit(matTrain_x.toarray(), oTrain_y)
oLR.fit(matTrain_x, oTrain_y)

#evaluating and comparing models.
#1. finding mean accuracy
print('SVC', oSVC.score(matTest_x, oTest_y))
print('DT', oDTC.score(matTest_x, oTest_y))
print('NB', oNB.score(matTest_x.toarray(), oTest_y))
print('LR', oLR.score(matTest_x, oTest_y))

#2. Confusion matrix
print('SVC', confusion_matrix(oTest_y, oSVC.predict(matTest_x), labels=['positive', 'negative']))
print('DT', confusion_matrix(oTest_y, oDTC.predict(matTest_x), labels=['positive', 'negative']))
print('NB', confusion_matrix(oTest_y, oNB.predict(matTest_x.toarray()), labels=['positive', 'negative']))
print('LR', confusion_matrix(oTest_y, oLR.predict(matTest_x), labels=['positive', 'negative']))

#Selecting SVC and further tuning the model
dictParameters = {'C': [1, 4, 8, 16, 32], 'kernel':['linear', 'rbf']}
oSVCTuned = GridSearchCV(oSVC, dictParameters, cv=5)
oSVCTuned.fit(matTrain_x, oTrain_y)

print('Tuned_SVC', oSVCTuned.score(matTest_x, oTest_y))

#outputting test results
print('---TEST OUTPUT---')
oSVCTuned.predict(matTest_x)
#print long string
pd.options.display.max_colwidth = 10000
#outputting results one by one along with review
for test in oTest_x:
    print(oSVCTuned.predict(oTfidfModel.transform([test])) , ' ', test)