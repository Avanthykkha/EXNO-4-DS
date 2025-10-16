# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

#feature scalling
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="340" height="230" alt="image" src="https://github.com/user-attachments/assets/b344edba-98e7-475a-bab0-2b7cfc980aed" />


```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
<img width="130" height="236" alt="image" src="https://github.com/user-attachments/assets/5af739ed-b61b-40ae-976f-26dd98081520" />


```
 df.dropna()
```
<img width="370" height="504" alt="image" src="https://github.com/user-attachments/assets/0df84949-f1d9-43d5-82ea-2827d3352448" />


```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="164" height="148" alt="image" src="https://github.com/user-attachments/assets/fad0889c-558c-432b-98ad-0f40672052fc" />



```
sc = StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="394" height="445" alt="image" src="https://github.com/user-attachments/assets/1f9712c4-b8cc-4cf3-b474-c150baee6802" />



```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Height', 'Weight']]=scaler.fit_transform(df[['Height', 'Weight']])
df.head()
```
<img width="376" height="255" alt="image" src="https://github.com/user-attachments/assets/50d1e737-9228-4c41-acf3-7fbaad437153" />


```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="384" height="515" alt="image" src="https://github.com/user-attachments/assets/2d77a17e-820c-4b5a-bc8e-7e6cbf9cffbf" />



```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="378" height="251" alt="image" src="https://github.com/user-attachments/assets/6f3b03cd-2c47-464e-8938-b6e80d8f1b22" />


```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="442" height="444" alt="image" src="https://github.com/user-attachments/assets/73bd378f-e069-4c68-9d8d-55522a698445" />


```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="193" height="587" alt="image" src="https://github.com/user-attachments/assets/a6414f08-c34d-477e-832c-33ce529094c2" />


```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1043" height="510" alt="image" src="https://github.com/user-attachments/assets/c38ab0fe-d741-45a0-bd07-99dced5d9482" />


```
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="901" height="507" alt="image" src="https://github.com/user-attachments/assets/ec13b93b-ef43-44ec-b432-a479b109e565" />


```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
<img width="403" height="98" alt="image" src="https://github.com/user-attachments/assets/b42c3c60-5980-4118-9659-6dfe0f3d11d9" />


```
 y_pred = rf.predict(X_test)
```
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="449" height="451" alt="image" src="https://github.com/user-attachments/assets/334eaabf-0a77-416c-8d0c-85a150372f06" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1043" height="514" alt="image" src="https://github.com/user-attachments/assets/3255914b-38c6-415f-9c9a-c12cbf1a46b1" />


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="907" height="503" alt="image" src="https://github.com/user-attachments/assets/ef260338-9580-40b8-913e-2a463baea0ce" />


```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="809" height="102" alt="image" src="https://github.com/user-attachments/assets/6d08e726-9f5f-45ce-a452-b9fba482cd68" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="441" height="87" alt="image" src="https://github.com/user-attachments/assets/f5d0162a-0567-40f5-a151-b217d86bc87e" />


```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
Model accuracy using selected features: 0.8242651657285803


```
!pip install skfeature-chappers
```
<img width="1518" height="357" alt="image" src="https://github.com/user-attachments/assets/7d9838f5-e724-4fb4-a088-9a85be799fec" />


```
 import numpy as np
 import pandas as pd
 from skfeature.function.similarity_based import fisher_score
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
```


```
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 df[categorical_columns] = df[categorical_columns].astype('category')
```


```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="916" height="511" alt="image" src="https://github.com/user-attachments/assets/75508845-0edf-421b-ba86-135b847ce0ff" />



```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```



```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```



```
selected_features_anova = X.columns[selector_anova.get_support()]
```


```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
Selected features using ANOVA:
Index(['age', 'relationship', 'gender', 'capitalgain', 'hoursperweek'], dtype='object')



```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
```



```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```


```
 df[categorical_columns]
```
<img width="912" height="492" alt="image" src="https://github.com/user-attachments/assets/45af6feb-ae48-4e41-857a-2c31500048d7" />


```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
```

```
logreg = LogisticRegression()
```


```
n_features_to_select =6
```


```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```


/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

  
  <img width="307" height="174" alt="image" src="https://github.com/user-attachments/assets/2b7ee69c-6fde-4b56-a612-dfc501e3fc62" />)



       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE
