# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:03:32 2020

@author: tanusha.goswami
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, scale, LabelBinarizer
from sklearn.feature_selection import chi2
from scipy import stats
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score,  classification_report
from pcm import plot_confusion_matrix

# Visualise pairwise plots
def plotData(data, column_names, colour_palette):
    col_names = column_names + ['Type_code']
    data = data[col_names]
    labels = np.sort(data['Type_code'].unique())
    colour_count = 0
    for encoders in labels:
        color = colour_palette[colour_count]
        colour_count += 1
        filtered_data = data.loc[data['Type_code'] == encoders]
#        print(filtered_data.size)
        plt.scatter(np.array(filtered_data.iloc[:,0]), np.array(filtered_data.iloc[:,1]), c = color)
#        plt.legend(labels)

def pairwise_plot(plot_features, data_encoded):
    pair_features = [[a, b] for a in plot_features  for b in plot_features if a != b]
    l = len(plot_features)
    colour_palette = []
    for i in plot_features:
        r = random.random()
        b = random.random()
        g = random.random()
        colour = np.array([r, g, b]).reshape(1,-1)
        colour_palette.append(colour)
        
    fig = plt.figure(figsize = (20,40))
    for i in range(1,len(pair_features)):
        col_name = pair_features[i-1]
        plt.subplot(l,1,i)
        plotData(data_encoded, col_name, colour_palette)
        plt.title(str(col_name))
        
def run_eval_logistic_regression(x_train, x_test, y_train, y_test, plot_pcm = False):
    logmodel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 5000)
    logmodel.fit(x_train, y_train)
    predictions = logmodel.predict(x_test)
    predictions_train = logmodel.predict(x_train)
    
    print('Accuracy of Training Model {:0.4f}'.format(accuracy_score(y_train, predictions_train)))
    print('Accuracy of Testing Model {:0.4f}'.format(accuracy_score(y_test, predictions))) 
    
    print('Classification Report: ')
    print(classification_report(y_test, predictions))
    if plot_pcm == True:
        print('Confusion Matrix of Test Data: ')
        plot_confusion_matrix(confusion_matrix(y_test, predictions),
                              target_names= sorted(y.unique()),
                              cmap=plt.cm.Blues,normalize=True)
        
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
    
data = pd.read_csv('pokemon_alopez247.csv')
# No data wrangling required

#plot_features = ['Total', 'HP', 'Attack','Defense', 'Sp_Atk', 
#                 'Sp_Def', 'Speed', 'Generation', 'isLegendary',
#                 'Color', 'hasGender', 'Pr_Male', 'Egg_Group_1',
#                 'hasMegaEvolution', 'Height_m', 'Weight_kg', 'Catch_Rate', 'Body_Style']
#

lb_make = LabelEncoder()
data["Type_code"] = lb_make.fit_transform(data["Type_1"])

#data_encoded = data.copy()
#data_encoded['Color'] = lb_make.fit_transform(data_encoded['Color'])
#data_encoded['Egg_Group_1'] = lb_make.fit_transform(data_encoded['Egg_Group_1'])
#data_encoded['Body_Style'] = lb_make.fit_transform(data_encoded['Body_Style'])
#data_encoded = data_encoded.fillna(-1)

# Most signficant continuos variables
data_no_nulls = data.loc[data['Pr_Male'].isnull() == False]
y = data_no_nulls.iloc[:,-1]
X = data_no_nulls.copy()
X = X.drop(columns = ['Number','Name','Type_1','Type_2','Egg_Group_2','Type_code', 'Color','Egg_Group_1','Body_Style'])
chi2_stats, p_values = chi2(X,y)

s = list(X.columns[(np.where(p_values < 0.05))[0]])
signficant_features = data[s]

# Most significant categorical variables
X = data.copy()
X = X.drop(columns = ['Number','Name','Type_1','Type_2','Egg_Group_2',
                      'Total', 'HP', 'Attack','Defense', 'Sp_Atk', 'Sp_Def', 
                      'Speed', 'Generation','isLegendary','hasGender', 
                      'Pr_Male','hasMegaEvolution','Height_m', 'Weight_kg', 
                      'Catch_Rate'])
    
cat_vars = X.columns.to_list()
cat_vars.remove('Type_code')

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X[var], prefix=var)
    data1=X.join(cat_list)
    X = data1
    
# Keeping only dummy variables and output
X = X.iloc[:, 3:]


chi2_list = p_values = np.zeros(X.shape[1])

for i in range(X.shape[1]):
    c = X.columns[i]
    cross_tab = pd.crosstab(X[c], X['Type_code'], margins = False) 
    chi2, p, dof, ex = stats.chi2_contingency(cross_tab, correction=False)
    chi2_list[i] = chi2
    p_values[i] = p

higher_cutoff = np.median(p_values)*100
# 0.05 increases variance of model
s_cat = list(X.columns[(np.where(p_values < higher_cutoff))[0]])
signficant_features_cat = X[s_cat]
s_cat.remove('Type_code')


data_for_lr = pd.concat([signficant_features, signficant_features_cat], axis = 1)
X = data_for_lr.copy()
y = data_for_lr['Type_code']
X = X.drop(columns = ['Type_code', 'Total'])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
run_eval_logistic_regression(x_train, x_test, y_train, y_test)

# Standardise Data 
# Improves model with higher_cutoff
x_train_std = scale(x_train)
x_test_std = scale(x_test)

run_eval_logistic_regression(x_train_std, x_test_std, y_train, y_test)







