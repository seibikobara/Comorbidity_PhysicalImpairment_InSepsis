

'''
This code is used to estimate an propensity score of attrition factors including death within 72 hours from ICU admission and death in a hospital after 72 hours from ICU admission using multiple machine learnin models 
'''




import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from fancyimpute import IterativeImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras




# data load
data = pd.read_csv("forpython.csv", index_col=[0])


# variables in use for imputation
numerical = [
    # baseline 
    "age_ICUadmission", "hp_days_log", "life_space_befo_ad_to_hp2",
    # conditions
    "APACHE2_ICU_admission", "SOFA_ICUadmission",
    "lactate_max", "EQ5D_vas_initial", "EQ5D_vas_last",
    "total_FIM_score_initial",  "total_FIM_score_last"
    ]
    
categorical = [
    # baseline
    "sex2", 
    "comor",  # because those vars are needed in a model as well as imputation
    "cardiopul", "cerebro", "metabolic", "cancer",
    # conditions
    "dnar_atleast",  "medical", "shock", "ARDS_bi", "AKI2",
    "blood_culture", "septic.encephalopathy",  "delirium",
    "septic.cardiomyopathy",
    # type_of_infection
    "type_of_infection_community_acquired",
    "type_of_infection_healthcare_associated",
    "type_of_infection_nosocomial",
    # admission route
    "ED_cli", "elective", "inpatients", "transf",
    # soi
    "respi2", "abdominal2", "uri2", "others2",  "unknown2",
    # intervention
    "RRT2", "MV",  "NIV",  "NHF",
    "propofol", "midazolam", "dex",
    "noradrenaline2", "hydrocortisone2", "vasopressin2",
    "CVC",
    "rehab_hp", "rehab_did_ICU",
    # discharge disposition
    "acute_care", "home", "home_health_services", "long_term", "others"]

bias = ['previous_dependence',"mortality_3day2", "hp_death_after_3days"]


# imputation set
impu_nu   = numerical
impu_cate = [categorical, bias]
explanatory = [impu_nu, impu_cate]



# baseline covariates that should be in a model
covariates = [
    # exposure variables
    "comor", "cardiopul", "cerebro", "metabolic", "cancer",
    # covariates in a model 
    "age_ICUadmission",
    "SOFA_ICUadmission",
    "sex2",
    "medical",  "shock",
    "respi2",
    "abdominal2",
    "uri2",
    "others2",
    "unknown2",
    'previous_dependence',
    "blood_culture"]

outcome = 'mortality_3day2' 
# outcome = 'hp_death_after_3days'

# separate numericals and categoricals for normalization only for numericals
covariates_nums = [ "age_ICUadmission", "SOFA_ICUadmission"]
covariates_cate = ["comor", "cardiopul", "cerebro", "metabolic", "cancer",
                    "sex2", "medical",  "shock",
                    "respi2","abdominal2","uri2","others2","unknown2",'previous_dependence',
                    "blood_culture"]


# multiple imputation
def mice_data(random_state_, data_):
    imputer = IterativeImputer(max_iter =1, random_state = random_state_)
    df = pd.DataFrame(imputer.fit_transform(data_))
    df.columns = data_.columns
    return df


# train test split
def split_normalize(x_mice,y_mice):
# split
    x_train, x_test, y_train, y_test = train_test_split(x_mice,y_mice)
    # normalize only numericals
    x_train_nu = x_train.loc[:,x_train.columns.isin(impu_nu)]
    x_train_cate = x_train.loc[:,~x_train.columns.isin(impu_nu)]
    x_test_nu = x_test.loc[:,x_test.columns.isin(impu_nu)]
    x_test_cate = x_test.loc[:,~x_test.columns.isin(impu_nu)]
    # normalize numericals
    mean_ = x_train_nu.mean(axis = 0)
    sd_   = x_train_nu.std(axis = 0)
    x_train_nu -= mean_
    x_train_nu /= sd_
    x_test_nu -= mean_
    x_test_nu /= sd_
    # concat
    x_train_fin = pd.concat([x_train_nu, x_train_cate], axis = 1)
    x_test_fin = pd.concat([x_test_nu, x_test_cate], axis = 1)
    return x_train_fin, x_test_fin, y_train, y_test



# Standardized mean difference calculation
# SMD for continuous
def smd_numeric(data , outcome, var):
    n,m = data.shape
    marge_P_outcome_1 = data[outcome].sum()/n
    marge_P_outcome_0 = 1- marge_P_outcome_1
    data.loc[data[outcome]==1, 'wts'] = marge_P_outcome_1/data.loc[data[outcome]==1, 'prob']
    data.loc[data[outcome]==0, 'wts'] = marge_P_outcome_0/(1-data.loc[data[outcome]==0, 'prob'])
    
    T = data.loc[data[outcome]==1, var]
    WT = data.loc[data[outcome]==1, 'wts']
    C = data.loc[data[outcome]==0, var]
    WC = data.loc[data[outcome]==0, 'wts']
    
    T_mean = T.mean()
    C_mean = C.mean()

    T_sd = T.std()
    C_sd = C.std()
    smd = (T_mean - C_mean)/(np.sqrt((T_sd**2 + C_sd**2)/2))

    wt_T_mean = np.sum(T*WT)/np.sum(WT)
    wt_C_mean = np.sum(C*WC)/np.sum(WC)
    wt_T_var  = np.sum(WT)/(np.sum(WT)**2 - np.sum(WT**2)) * np.sum(WT*(T-wt_T_mean)**2) 
    wt_C_var  = np.sum(WC)/(np.sum(WC)**2 - np.sum(WC**2)) * np.sum(WC*(C-wt_C_mean)**2)
    wt_smd = (wt_T_mean - wt_C_mean)/np.sqrt((wt_T_var + wt_C_var)/2)

    return np.absolute(smd), np.absolute(wt_smd)

# SMD for categorical
def smd_categorical(data , outcome, var):
    n,m = data.shape
    marge_P_outcome_1 = data[outcome].sum()/n
    marge_P_outcome_0 = 1- marge_P_outcome_1
    data.loc[data[outcome]==1, 'wts'] = marge_P_outcome_1/data.loc[data[outcome]==1, 'prob']
    data.loc[data[outcome]==0, 'wts'] = marge_P_outcome_0/(1-data.loc[data[outcome]==0, 'prob'])
    
    T = data.loc[data[outcome]==1, var]
    WT = data.loc[data[outcome]==1, 'wts']
    C = data.loc[data[outcome]==0, var]
    WC = data.loc[data[outcome]==0, 'wts']
    
    p1 = np.sum(T)/len(T)
    p2 = np.sum(C)/len(C)
    smd = (p1-p2)/np.sqrt((p1*(1-p1) + p2*(1-p2))/2)
    
    wt_p1 = np.sum(T*WT)/np.sum(WT*1)
    wt_p2 = np.sum(C*WC)/np.sum(WC*1)
    
    wt_p1_var = wt_p1*(1-wt_p1)
    wt_p2_var = wt_p2*(1-wt_p2)

    wt_smd = (wt_p1 - wt_p2)/np.sqrt((wt_p1_var + wt_p2_var)/2)
    
    return np.absolute(smd), np.absolute(wt_smd)



#-----------------------------
# logitic regression (no optimization)
#-----------------------------

roc_list = list()
smd_list = list()
for i in np.arange(1,11):
    df = mice_data(i, data)
    x_mice = df[covariates]
    y_mice= df[outcome]
    # split
    x_train, x_test, y_train, y_test = split_normalize(x_mice, y_mice)
    # combine
    x = pd.concat([x_train, x_test], axis = 0)
    y = pd.concat([y_train, y_test], axis = 0)
    # smote
    smote = SMOTE(random_state = 1, k_neighbors=3)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    # logistic regression 
    lrc_eva = LogisticRegression(penalty='l2', C = 1)
    lrc_eva.fit(x_resampled, y_resampled)   
    # ROC
    y_prob = lrc_eva.predict_proba(x)[:,1]
    y_true = y
    roc = roc_auc_score(y_true, y_prob)
    roc_list = roc_list + [roc]
    # SMD in test set
    temp = pd.concat([x, y],axis =1)
    temp.loc[:,'prob'] = y_prob
    for i in covariates_nums:
        a,b =  smd_numeric(temp, outcome,i)
        res_ = [i,a,b]
        smd_list = smd_list + [res_]
    for j in covariates_cate:
        c,d = smd_categorical(temp, outcome,j)
        res_ = [j, c, d]
        smd_list = smd_list + [res_]


# check
smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
res = smd.groupby('var').mean().reset_index()

# summarize
res.to_csv('smd_log.txt')
pd.DataFrame(roc_list).to_csv('roc_log.txt')



#-----------------------------
# Gradient boosting decision tree
#-----------------------------
# optimization
track = list()
for k in [5,10,20]:       # max_depth
    for l in [0.001,0.01]: # learning rate
        for m in [10,100,150,200,250,300, 350, 400]: # n tree
            for n in np.arange(1,11): # imp
                df = mice_data(n, data)
                x_mice = df[covariates]
                y_mice= df[outcome]
                skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
                roc_list = list()
                smd_list = list()
                for train_index, test_index in skf.split(x_mice, y_mice):
                    x_train_cv, x_test_cv = x_mice.iloc[train_index], x_mice.iloc[test_index]
                    y_train_cv, y_test_cv = y_mice.iloc[train_index], y_mice.iloc[test_index]
                    # smote
                    smote = SMOTE(random_state = 1, k_neighbors=3)
                    x_train_cv_resampled, y_train_cv_resampled = smote.fit_resample(x_train_cv, y_train_cv)
                    # XGBoosted
                    xgbc = xgb.XGBClassifier(n_estimators =m, eta = l, max_depth = k)
                    xgbc.fit(x_train_cv_resampled, y_train_cv_resampled)
                    # ROC
                    y_prob = xgbc.predict_proba(x_test_cv)[:,1]
                    y_true = y_test_cv
                    roc = roc_auc_score(y_true, y_prob)
                    roc_list = roc_list + [roc]
                    # SMD in test set
                    temp = pd.concat([x_test_cv, y_test_cv],axis =1)
                    temp.loc[:,'prob'] = y_prob
                    for i in covariates_nums:
                        a,b =  smd_numeric(temp, outcome,i)
                        res_ = [i,a,b]
                        smd_list = smd_list + [res_]
                    for j in covariates_cate:
                        c,d = smd_categorical(temp, outcome,j)
                        res_ = [j, c, d]
                        smd_list = smd_list + [res_]
                # summarize
                smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
                smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
                mean_roc = np.mean(roc_list)
                mean_smd_diff = smd['smd_wt_smd'].mean()
                track_ = [k,l,m,n , mean_roc, mean_smd_diff]
                track = track + [track_]

# summarize
temp = pd.DataFrame(track, columns = ['depth','lrate','ntree','imp','roc','diff_smd'])
temp['para'] = 'Depth: '+ temp['depth'].astype(str)+', lrate: '+temp['lrate'].astype(str) + ', ntree: ' + temp['ntree'].astype(str)
res = temp.groupby('para').mean().reset_index()

# visualize
res = res.sort_values(by =['ntree'])
fig, ax = plt.subplots(1,1,figsize=(25,25), dpi =70)
ax = plt.plot(res['para'], res['roc'], c='red', label = 'AUROC')
ax = plt.plot(res['para'], res['diff_smd'], c='blue' ,label = 'Averaged difference in SMD')
ax = plt.xticks(rotation=90, ha='right')
ax = plt.legend()
ax = plt.ylabel('AUROC and SMD')
plt.savefig('tuning_xgb.pdf')




# XGBoost performance evaluation
roc_list = list()
smd_list = list()
for i in np.arange(1,11):
    df = mice_data(i, data)
    x_mice = df[covariates]
    y_mice= df[outcome]
    # smote
    smote = SMOTE(random_state = 1, k_neighbors=3)
    x_mice_resampled, y_mice_resampled = smote.fit_resample(x_mice, y_mice)
    # XGBoosted
    xgbc = xgb.XGBClassifier(n_estimators =100, eta = 0.01, max_depth = 20)
    xgbc.fit(x_mice_resampled, y_mice_resampled)
    # ROC
    y_prob = xgbc.predict_proba(x_mice)[:,1]
    y_true = y_mice
    roc = roc_auc_score(y_true, y_prob)
    roc_list = roc_list + [roc]
    # SMD in test set
    temp = pd.concat([x_mice, y_mice],axis =1)
    temp.loc[:,'prob'] = y_prob
    for i in covariates_nums:
        a,b =  smd_numeric(temp, outcome,i)
        res_ = [i,a,b]
        smd_list = smd_list + [res_]
    for j in covariates_cate:
        c,d = smd_categorical(temp, outcome,j)
        res_ = list()
        res_ = [j, c, d]
        smd_list = smd_list + [res_]

smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
res = smd.groupby('var').mean().reset_index()

# summarise
res.to_csv('~/projects/sepsis_2020/ml_exclude_test_set/results/smd_xgb.txt')
pd.DataFrame(roc_list).to_csv('~/projects/sepsis_2020/ml_exclude_test_set/results/roc_xgb.txt')


#--------------------------------
# neural network model
#--------------------------------
# NN optimization
# hyperparameters of neurons.
track = list()  
for i in [10, 15, 20]: # neuron1
    for j in [2, 5, 10]:  # neuron2
        #for h in [10.15]:  # n3
        for k in np.arange(1,11,1): # imp            
            df = mice_data(1, data)
            x_mice = df[covariates]
            y_mice= df[outcome]
            # split
            x_train, x_test, y_train, y_test = split_normalize(x_mice, y_mice)
            # combine
            x = pd.concat([x_train, x_test], axis =0)
            y = pd.concat([y_train, y_test], axis =0)
            skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
            roc_list = list()
            smd_list = list()
            for train_index, test_index in skf.split(x, y):
                x_train_cv, x_test_cv = x.iloc[train_index], x.iloc[test_index]
                y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
                # smote
                smote = SMOTE(random_state= 1, k_neighbors=3)
                x_train_cv_resampled, y_train_cv_resampled = smote.fit_resample(x_train_cv, y_train_cv)
                # NN
                model=keras.Sequential([
                keras.layers.Dense(units=i,activation='relu'),
                keras.layers.Dense(units=j,activation='relu'),
                #keras.layers.Dense(units=h,activation='relu'),
                keras.layers.Dense(units=1, activation='sigmoid')])
                model.compile(loss='binary_crossentropy', optimizer= 'adam')
                model.fit(x_train_cv_resampled, y_train_cv_resampled,epochs=5)
                # roc
                y_prob = model.predict(x_test_cv)[:, 0] 
                y_true = y_test_cv
                roc = roc_auc_score(y_true, y_prob)
                roc_list = roc_list + [roc]
                # SMD in test set
                temp = pd.concat([x_test_cv, y_test_cv],axis =1)
                temp.loc[:,'prob'] = y_prob
                for l in covariates_nums:
                    a,b =  smd_numeric(temp, outcome,l)
                    res_ = [l,a,b]
                    smd_list = smd_list + [res_]
                for m in covariates_cate:
                    c,d = smd_categorical(temp, outcome,m)
                    res_ = [m, c,d]
                    smd_list = smd_list + [res_]
        # summarize
        smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
        smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
        mean_roc = np.mean(roc_list)
        mean_smd_diff = smd['smd_wt_smd'].mean()
        track_ = [i, j, k, mean_roc, mean_smd_diff]
        track = track + [track_]
        
temp = pd.DataFrame(track, columns = ['n1', 'n2','imp', 'roc','diff_smd'])
temp['para'] = 'N1: ' + temp['n1'].astype(str) + ', n2: ' + temp['n2'].astype(str) 
res = temp.groupby('para').mean().reset_index()

# visualize
res = res.sort_values(by=['n1'])
fig, ax  = plt.subplots(1,1, figsize =(12,12), dpi=72)
ax = plt.plot(res['para'], res['roc'], c= 'red', label = 'AUROC')
ax = plt.plot(res['para'],res['diff_smd'], c = 'blue', label = 'Averaged difference in SMD')
ax = plt.xticks(rotation = 90, ha='right')
ax = plt.legend()
ax = plt.ylabel('AUROC and SMD')
plt.savefig('tuning_nn.pdf') 


# tuning epoch
track = list()  
for h in [10,50,100,150, 200]:  #epoch
    for k in np.arange(1,11,1): # imp            
        df = mice_data(1, data)
        x_mice = df[covariates]
        y_mice= df[outcome]
        # split
        x_train, x_test, y_train, y_test = split_normalize(x_mice, y_mice)
        # combine
        x = pd.concat([x_train, x_test], axis = 0)
        y = pd.concat([y_train, y_test], axis = 0)
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
        roc_list = list()
        smd_list = list()
        for train_index, test_index in skf.split(x, y):
            x_train_cv, x_test_cv = x.iloc[train_index], x.iloc[test_index]
            y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
            # smote
            smote = SMOTE(random_state= 1, k_neighbors=3)
            x_train_cv_resampled, y_train_cv_resampled = smote.fit_resample(x_train_cv, y_train_cv)
            # NN
            model=keras.Sequential([
            keras.layers.Dense(units=20,activation='relu'),
            keras.layers.Dense(units=5,activation='relu'),
            keras.layers.Dense(units=1, activation='sigmoid')])
            model.compile(loss='binary_crossentropy', optimizer= 'adam')
            model.fit(x_train_cv_resampled, y_train_cv_resampled,epochs=h)
            # roc
            y_prob = model.predict(x_test_cv)[:, 0] 
            y_true = y_test_cv
            roc = roc_auc_score(y_true, y_prob)
            roc_list = roc_list + [roc]
            # SMD in test set
            temp = pd.concat([x_test_cv, y_test_cv],axis =1)
            temp.loc[:,'prob'] = y_prob
            for l in covariates_nums:
                a,b =  smd_numeric(temp, outcome,l)
                res_ = [l,a,b]
                smd_list = smd_list + [res_]
            for m in covariates_cate:
                c,d = smd_categorical(temp, outcome,m)
                res_ = [m, c, d]
                smd_list = smd_list + [res_]
    # summarize
    smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
    smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
    mean_roc = np.mean(roc_list)
    mean_smd_diff = smd['smd_wt_smd'].mean()
    track_ = [h, k,mean_roc, mean_smd_diff]
    track = track + [track_]
temp = pd.DataFrame(track, columns = ['epoch','imp', 'roc','diff_smd'])
temp['para'] = 'Epoch: ' + temp['epoch'].astype(str) 
res = temp.groupby('para').mean().reset_index()

# visualize
res = res.sort_values(by= ['epoch'])
fig, ax  = plt.subplots(1,1, figsize =(12,12), dpi=72)
ax = plt.plot(res['para'], res['roc'], c= 'red', label = 'AUROC')
ax = plt.plot(res['para'],res['diff_smd'], c = 'blue', label = 'Averaged difference in SMD')
ax = plt.xticks(rotation = 90, ha='right')
ax = plt.legend()
ax = plt.ylabel('AUROC and SMD')
ax = plt.ylim(-5,3)
plt.savefig('tuning_nn_epoch.pdf') 



# model performance 
roc_list = list()
smd_list = list()
for i in np.arange(1,11,1):
    df = mice_data(i, data)
    x_mice = df[covariates]
    y_mice= df[outcome]
    # split
    x_train, x_test, y_train, y_test = split_normalize(x_mice, y_mice)
    x = pd.concat([x_train, x_test], axis = 0)
    y = pd.concat([y_train, y_test], axis = 0)
    # smote
    smote = SMOTE(random_state= 1, k_neighbors=3)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    # NN
    model=keras.Sequential([
    keras.layers.Dense(units=20,activation='relu'),
    keras.layers.Dense(units=5,activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer= 'adam')
    model.fit(x_resampled, y_resampled,epochs=10)
    # roc
    y_prob = model.predict(x)[:, 0] 
    y_true = y
    roc = roc_auc_score(y_true, y_prob)
    roc_list = roc_list + [roc]
    # SMD in test set
    temp = pd.concat([x, y],axis =1)
    temp.loc[:,'prob'] = y_prob
    for i in covariates_nums:
        a,b =  smd_numeric(temp, outcome,i)
        res_ = [i,a,b]
        smd_list = smd_list + [res_]
    for j in covariates_cate:
        c,d = smd_categorical(temp, outcome,j)
        res_ = [j, c,d]
        smd_list = smd_list + [res_]

# summarize
smd = pd.DataFrame(smd_list, columns = ['var','smd','wt_smd'])
smd['smd_wt_smd'] = smd['smd']  - smd['wt_smd']
res = smd.groupby('var').mean().reset_index()

res.to_csv('smd_nn.txt')
pd.DataFrame(roc_list).to_csv('roc_nn.txt')