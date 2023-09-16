import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek


def plot_data(X,y,ax,title):
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, s = 30, edgecolor=(0,0,0,0.5))
    ax.set_ylabel('Principle Component 1')
    ax.set_xlabel('Principle Component 2')
    if title is not None:
        ax.set_title(title)

def reShapeArray(original_array,new_size):
    new_array = np.empty(new_size) 
    new_array.fill(None)
    new_array[:original_array.size] = original_array
    return new_array

def add_to_csv(label_old,label_new,x_train,name):
    size = max(label_old.size,x_train.shape[0])
    label_old = reShapeArray(label_old,size)
    label_new = reShapeArray(label_new,size)
    data = {
        'Predicted labels before feature engineering': label_old,
        'Predicted labels after feature engineering': label_new,
        'No of new features': [x_train.shape[1]]*(size)
    }
    df = pd.DataFrame(data)
    for i in range(x_train.shape[1]):
        new_feature_name = f'new_feature_{i + 1}'
        df[new_feature_name] = x_train[:, i]
    df.to_csv(name, index=False)
    return df

# Read column names from file
cols = list(pd.read_csv("train.csv", nrows=1))      

# Use list comprehension to remove the unwanted column in **usecol**
train_data= pd.read_csv("train.csv", usecols =[i for i in cols if i not in ["label_2", "label_3", "label_4"]])
train_data = train_data.dropna()
train_data.head()

test_data= pd.read_csv("valid.csv", usecols =[i for i in cols if i not in ["label_2", "label_3", "label_4"]])
test_data = test_data.dropna()
test_data.head()

test_data_= pd.read_csv("test.csv", usecols =[i for i in cols if i not in ["label_1","label_2", "label_3", "label_4"]])
test_data_ = test_data_.dropna()
X1_test_ = test_data_.iloc[:,:]

#features
X1_train = train_data.iloc[:,:-1]
Y1_train = train_data.iloc[:,-1]
X1_test = test_data.iloc[:,:-1]
Y1_test = test_data.iloc[:,-1]

rf = RandomForestClassifier()
rf.fit(X1_train, Y1_train)

Y1_pred_rf = rf.predict(X1_test)
print(classification_report(Y1_test, Y1_pred_rf))

Y1_pred_rf_ = rf.predict(X1_test_)

#Observe correlation between features
X1_train.corr()

pca = PCA(0.98)
pca = pca.fit(X1_train)

X1_train_pca = pca.transform(X1_train)
X1_test_pca = pca.transform(X1_test)
X1_test_pca_ = pca.transform(X1_test_)
X1_train_pca.shape

train_data['label_1'].value_counts().plot(kind='bar',title='Count of Label_1')

rf = RandomForestClassifier()
rf.fit(X1_train_pca, Y1_train)
Y1_pred_rf = rf.predict(X1_test_pca)

importance = rf.feature_importances_
columns_to_delete = []
for i,v in enumerate(importance):
    if v < 0.008:
        columns_to_delete.append(i)   
train_reduced = np.delete(X1_train_pca, columns_to_delete, axis=1)
test_reduced = np.delete(X1_test_pca, columns_to_delete, axis=1)
test_reduced_ = np.delete(X1_test_pca_, columns_to_delete, axis=1)
train_reduced.shape

rf = RandomForestClassifier()
rf.fit(train_reduced, Y1_train)
Y1_pred = rf.predict(test_reduced)
print(classification_report(Y1_test, Y1_pred))

Y1_pred_ = rf.predict(test_reduced_)

add_to_csv(Y1_pred_rf_,Y1_pred_,test_reduced_,'190116U_label_1.csv').head()

# Use list comprehension to remove the unwanted column in **usecol**
train_data= pd.read_csv("train.csv", usecols =[i for i in cols if i not in ["label_1", "label_3", "label_4"]])
train_data = train_data.dropna()
train_data.head()

test_data= pd.read_csv("valid.csv", usecols =[i for i in cols if i not in ["label_1", "label_3", "label_4"]])
test_data = test_data.dropna()
test_data.head()

test_data_= pd.read_csv("test.csv", usecols =[i for i in cols if i not in ["label_1","label_2", "label_3", "label_4"]])
test_data_ = test_data_.dropna()
X2_test_ = test_data_.iloc[:,:]

#features
X2_train = train_data.iloc[:,:-1]
Y2_train = train_data.iloc[:,-1]
X2_test = test_data.iloc[:,:-1]
Y2_test = test_data.iloc[:,-1]

#Feature Scaling
scaler = StandardScaler()
scaler.fit(X2_train)

X2_train_sca = scaler.transform(X2_train)
X2_test_sca = scaler.transform(X2_test)
X2_test_sca_ = scaler.transform(X2_test_)

xgb_regressor = xgb.XGBRegressor()
xgb_regressor.fit(X2_train_sca,Y2_train)

Y2_pred = xgb_regressor.predict(X2_test_sca)
print(f"mean squared error: {mean_squared_error(Y2_test,Y2_pred)}")

Y2_pred_ = xgb_regressor.predict(X2_test_sca_)

#Observe correlation between features
X2_train.corr()

pca = PCA(0.95)
pca = pca.fit(X2_train)

X2_train_pca = pca.transform(X2_train)
X2_test_pca = pca.transform(X2_test)
X2_test_pca_ = pca.transform(X2_test_)
X2_train_pca.shape

#Feature Scaling
scaler = StandardScaler()
scaler.fit(X2_train_pca)

X2_train_pca_sca = scaler.transform(X2_train_pca)
X2_test_pca_sca = scaler.transform(X2_test_pca)
X2_test_pca_sca_ = scaler.transform(X2_test_pca_)

xgb_regressor = xgb.XGBRegressor()
xgb_regressor.fit(X2_train_pca_sca,Y2_train)

Y2_pred_new = xgb_regressor.predict(X2_test_pca_sca)
print(f"mean squared error: {mean_squared_error(Y2_test,Y2_pred_new)}")

Y2_pred_new_ = xgb_regressor.predict(X2_test_pca_sca_)

add_to_csv(Y2_pred_,Y2_pred_new_,X2_test_pca_sca_,'190116U_label_2.csv').head()

# Use list comprehension to remove the unwanted column in **usecol**
train_data= pd.read_csv("train.csv", usecols =[i for i in cols if i not in ["label_1", "label_2", "label_4"]])
train_data = train_data.dropna()
train_data.head()

test_data= pd.read_csv("valid.csv", usecols =[i for i in cols if i not in ["label_1", "label_2", "label_4"]])
test_data = test_data.dropna()
test_data.head()

test_data_= pd.read_csv("test.csv", usecols =[i for i in cols if i not in ["label_1", "label_2","label_3","label_4"]])
test_data_ = test_data_.dropna()
test_data_.head()

#features
X3_train = train_data.iloc[:,:-1]
Y3_train = train_data.iloc[:,-1]
X3_test = test_data.iloc[:,:-1]
Y3_test = test_data.iloc[:,-1]
X3_test_ = test_data_.iloc[:,:]

rf = RandomForestClassifier()
rf.fit(X3_train, Y3_train)
Y3_pred_rf = rf.predict(X3_test)
print(classification_report(Y3_test, Y3_pred_rf))

Y3_pred_rf_ = rf.predict(X3_test_)

#Observe correlation between features
X3_train.corr()

pca = PCA(0.98)
pca = pca.fit(X3_train)

X3_train_pca = pca.transform(X3_train)
X3_test_pca = pca.transform(X3_test)
X3_test_pca_ = pca.transform(X3_test_)
X3_train_pca.shape

train_data['label_3'].value_counts().plot(kind='bar',title='Count of Label_3')



fig,ax = plt.subplots(figsize=(5, 5))
plot_data(X3_train_pca, Y3_train, ax, title='Original Dataset')


# Perform random sampling
smotetomek = SMOTETomek(random_state=0)
X3_train_pca, Y3_train = smotetomek.fit_resample(X3_train_pca, Y3_train)
fig,ax = plt.subplots(figsize=(5, 5))
plot_data(X3_train_pca, Y3_train, ax, title='Unbaised Dataset')
print(X3_train_pca.shape)

Y3_train.value_counts().plot(kind='bar',title='Count of Label_3')

rf = RandomForestClassifier()
rf.fit(X3_train_pca, Y3_train)
Y3_pred_rf = rf.predict(X3_test_pca)
print(classification_report(Y3_test, Y3_pred_rf))

importance = rf.feature_importances_
columns_to_delete = []
for i,v in enumerate(importance):
    if v < 0.008:
        columns_to_delete.append(i)   
train_reduced = np.delete(X3_train_pca, columns_to_delete, axis=1)
test_reduced = np.delete(X3_test_pca, columns_to_delete, axis=1)
test_reduced_ = np.delete(X3_test_pca_, columns_to_delete, axis=1)
train_reduced.shape

rf = RandomForestClassifier()
rf.fit(train_reduced, Y3_train)
Y3_pred_rf_new = rf.predict(test_reduced)
print(classification_report(Y3_test, Y3_pred_rf_new))

Y3_pred_rf_new_ = rf.predict(test_reduced_)

add_to_csv(Y3_pred_rf_,Y3_pred_rf_new_,test_reduced_,'190116U_label_3.csv').head()

# Use list comprehension to remove the unwanted column in **usecol**
train_data= pd.read_csv("train.csv", usecols =[i for i in cols if i not in ["label_1", "label_2", "label_3"]])
train_data = train_data.dropna()
train_data.head()

test_data= pd.read_csv("valid.csv", usecols =[i for i in cols if i not in ["label_1", "label_2", "label_3"]])
test_data = test_data.dropna()
test_data.head()

test_data_= pd.read_csv("test.csv", usecols =[i for i in cols if i not in ["label_1", "label_2", "label_3", "label_4"]])
test_data_ = test_data_.dropna()
test_data_.head()

#features
X4_train = train_data.iloc[:,:-1]
Y4_train = train_data.iloc[:,-1]
X4_test = test_data.iloc[:,:-1]
Y4_test = test_data.iloc[:,-1]
X4_test_ = test_data_.iloc[:,:]

rf = RandomForestClassifier()
rf.fit(X4_train, Y4_train)

Y4_pred_rf = rf.predict(X4_test)
print(classification_report(Y4_test, Y4_pred_rf))

Y4_pred_rf_ = rf.predict(X4_test_)

#Observe correlation between features
X4_train.corr()

pca = PCA(0.98)
pca = pca.fit(X4_train)

X4_train_pca = pca.transform(X4_train)
X4_test_pca = pca.transform(X4_test)
X4_test_pca_ = pca.transform(X4_test_)
X4_train_pca.shape

train_data['label_4'].value_counts().plot(kind='bar',title='Count of Label_4')

def plot_data(X,y,ax,title):
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, s = 30, edgecolor=(0,0,0,0.5))
    ax.set_ylabel('Principle Component 1')
    ax.set_xlabel('Principle Component 2')
    if title is not None:
        ax.set_title(title)
fig,ax = plt.subplots(figsize=(5, 5))
plot_data(X4_train_pca, Y4_train, ax, title='Original Dataset')

# Perform random sampling
smotetomek = SMOTETomek(random_state=0)
X4_train_pca, Y4_train = smotetomek.fit_resample(X4_train_pca, Y4_train)
fig,ax = plt.subplots(figsize=(5, 5))
plot_data(X4_train_pca, Y4_train, ax, title='Unbaised Dataset')
print(X4_train_pca.shape)

Y4_train.value_counts().plot(kind='bar',title='Count of Label_4')

rf = RandomForestClassifier()
rf.fit(X4_train_pca, Y4_train)
Y4_pred_rf = rf.predict(X4_test_pca)
print(classification_report(Y4_test, Y4_pred_rf))

importance = rf.feature_importances_
columns_to_delete = []
for i,v in enumerate(importance):
    if v < 0.008:
        columns_to_delete.append(i)   
train_reduced = np.delete(X4_train_pca, columns_to_delete, axis=1)
test_reduced = np.delete(X4_test_pca, columns_to_delete, axis=1)
test_reduced_ = np.delete(X4_test_pca_, columns_to_delete, axis=1)
train_reduced.shape

rf = RandomForestClassifier()
rf.fit(train_reduced, Y4_train)
Y4_pred_rf_new = rf.predict(test_reduced)
print(classification_report(Y4_test, Y4_pred_rf_new))

Y4_pred_rf_new_ = rf.predict(test_reduced_)

add_to_csv(Y4_pred_rf_,Y4_pred_rf_new_,test_reduced_,'190116U_label_4.csv').head()