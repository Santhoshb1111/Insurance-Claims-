#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pip uninstall numpy pandas -y


# In[3]:


pip install numpy pandas


# In[4]:


pip install numpy==1.24.4 pandas==1.5.3


# In[5]:


import numpy
import pandas

print("Numpy version:", numpy.__version__)
print("Pandas version:", pandas.__version__)


# In[6]:


pip install pandas


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


get_ipython().system('pip uninstall -y numpy pandas')


# In[9]:


get_ipython().system('pip install numpy==1.24.4 pandas==1.5.3')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


import pandas as pd


# In[ ]:


get_ipython().system('pip install --upgrade --force-reinstall numpy==1.24.4 pandas==1.5.3')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
import shap


# In[3]:


get_ipython().system('pip install shap')


# In[4]:


from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
import shap


# In[5]:


import sys
get_ipython().system('{sys.executable} -m pip install shap')


# In[6]:


pip install --upgrade pip


# In[1]:


from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error
import shap


# In[2]:


df = pd.read_csv('insurance_claims.csv')


# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('insurance_claims.csv')


# In[5]:


# 2️⃣ Basic EDA
print(df.shape, df.dtypes)
display(df.head())
sns.countplot(data=df, x='incident_type'); plt.title('Incidents by Type'); plt.show()
sns.histplot(df['total_claim_amount'], bins=50, kde=True); plt.title('Claim Amount Distribution');


# In[6]:


import seaborn as sns


# In[7]:


print(df.shape, df.dtypes)
display(df.head())
sns.countplot(data=df, x='incident_type'); plt.title('Incidents by Type'); plt.show()
sns.histplot(df['total_claim_amount'], bins=50, kde=True); plt.title('Claim Amount Distribution');


# In[8]:


import matplotlib.pyplot as plt


# In[10]:


print(df.shape, df.dtypes)
display(df.head(10))
sns.countplot(data=df, x='incident_type'); plt.title('Incidents by Type'); plt.show()
sns.histplot(df['total_claim_amount'], bins=50, kde=True); plt.title('Claim Amount Distribution');


# In[11]:


# 3️⃣ Feature Engineering
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['days_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df['bumper_to_bumper'] = df['vehicle_damage'].map({'Yes':1,'No':0})


# In[12]:


# 3️⃣ Feature Engineering
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['days_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df['bumper_to_bumper'] = df['property_damage'].map({'Yes':1,'No':0})


# In[13]:


# 4️⃣ Preprocessing & Pipeline Setup
numeric_feats = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_feats = df.select_dtypes(include=['object']).drop(['incident_type','fraud_reported'], axis=1).columns.tolist()

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='N/A')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])


# In[14]:


# 5️⃣ Target Definitions
y_clf = LabelEncoder().fit_transform(df['incident_type'])
y_reg = df['total_claim_amount']
X = df.drop(['incident_type','fraud_reported','total_claim_amount','policy_number'], axis=1)

X_train, X_test, yclf_train, yclf_test = train_test_split(X, y_clf, stratify=y_clf, test_size=0.2, random_state=42)
_, _, yreg_train, yreg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)


# In[15]:


# 6️⃣ Classification Pipeline
clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
clf_params = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20]}
clf_search = GridSearchCV(clf_pipeline, clf_params, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
clf_search.fit(X_train, yclf_train)

print("Best classifier ROC-AUC:", clf_search.best_score_)
ypred_proba = clf_search.predict_proba(X_test)[:,1]
print(classification_report(yclf_test, clf_search.predict(X_test)))
print("ROC-AUC on test:", roc_auc_score(yclf_test, ypred_proba))

# SHAP Feature Importance
explainer = shap.TreeExplainer(clf_search.best_estimator_.named_steps['clf'])
X_sample = preprocessor.transform(X_train.sample(200, random_state=1))
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=numeric_feats+list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_feats)))


# In[16]:


expected_cols = ['age', 'policy_annual_premium', 'incident_type']  # Example
missing_cols = [col for col in expected_cols if col not in X_train.columns]

if missing_cols:
    print("⚠️ Missing columns:", missing_cols)
else:
    print("✅ All good")


# In[17]:


# 6️⃣ Classification Pipeline
clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
clf_params = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20]}
clf_search = GridSearchCV(clf_pipeline, clf_params, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
clf_search.fit(X_train, yclf_train)

print("Best classifier ROC-AUC:", clf_search.best_score_)
ypred_proba = clf_search.predict_proba(X_test)[:,1]
print(classification_report(yclf_test, clf_search.predict(X_test)))
print("ROC-AUC on test:", roc_auc_score(yclf_test, ypred_proba))

# SHAP Feature Importance
explainer = shap.TreeExplainer(clf_search.best_estimator_.named_steps['clf'])
X_sample = preprocessor.transform(X_train.sample(200, random_state=1))
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=numeric_feats+list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_feats)))

# 7️⃣ Regression Pipeline


# In[18]:


print(X_train.columns.tolist())


# In[19]:


ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age', 'policy_number'])  # ❌ policy_number missing
])


# In[20]:


ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['age'])  # ✅ Remove or replace 'policy_number'
])


# In[21]:


feature_cols = df.columns.drop('fraud_reported')  # or similar


# In[22]:


feature_cols = [col for col in feature_cols if col != 'policy_number']


# In[23]:


X = df[feature_cols]


# In[24]:


clf_search.fit(X_train, yclf_train)


# In[25]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ['age', 'policy_annual_premium']  # ❌ problem here
cat_features = ['insured_sex', 'insured_education_level']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)


# In[26]:


# 6️⃣ Classification Pipeline
clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
clf_params = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20]}
clf_search = GridSearchCV(clf_pipeline, clf_params, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
clf_search.fit(X_train, yclf_train)

print("Best classifier ROC-AUC:", clf_search.best_score_)
ypred_proba = clf_search.predict_proba(X_test)[:,1]
print(classification_report(yclf_test, clf_search.predict(X_test)))
print("ROC-AUC on test:", roc_auc_score(yclf_test, ypred_proba))

# SHAP Feature Importance
explainer = shap.TreeExplainer(clf_search.best_estimator_.named_steps['clf'])
X_sample = preprocessor.transform(X_train.sample(200, random_state=1))
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=numeric_feats+list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_feats)))

# 7️⃣ Regression Pipeline


# In[27]:


print(yclf_test.unique())


# In[28]:


import pandas as pd
print(pd.Series(yclf_test).unique())


# In[29]:


clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
clf_params = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20]}
clf_search = GridSearchCV(clf_pipeline, clf_params, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
clf_search.fit(X_train, yclf_train)

print("Best classifier ROC-AUC:", clf_search.best_score_)
ypred_proba = clf_search.predict_proba(X_test)[:,1]
print(classification_report(yclf_test, clf_search.predict(X_test)))
print("ROC-AUC on test:", roc_auc_score(yclf_test, ypred_proba))

# SHAP Feature Importance
explainer = shap.TreeExplainer(clf_search.best_estimator_.named_steps['clf'])
X_sample = preprocessor.transform(X_train.sample(200, random_state=1))
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, feature_names=numeric_feats+list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_feats)))

# 7️⃣ Regression Pipeline


# In[30]:


reg_pipeline = Pipeline([
    ('prep', preprocessor),
    ('reg', GradientBoostingRegressor(random_state=42))
])
reg_params = {'reg__n_estimators':[100,200], 'reg__learning_rate':[0.05,0.1]}
reg_search = GridSearchCV(reg_pipeline, reg_params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
reg_search.fit(X_train, yreg_train)

print("Best regressor RMSE:", -reg_search.best_score_)
yreg_pred = reg_search.predict(X_test)
print("Test RMSE:", mean_squared_error(yreg_test, yreg_pred, squared=False))


# In[31]:


sns.lineplot(x=yreg_test, y=yreg_pred); plt.xlabel('Actual Claim'); plt.ylabel('Predicted Claim'); plt.title('Regression Fit'); plt.show()


# In[ ]:




