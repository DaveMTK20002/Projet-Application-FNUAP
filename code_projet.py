## Implémentation des différents modèles

##Modules de traitement
import numpy as np #pour les vecteurs, array, uni-dimensionnel
import pandas as pd #pour les dataframes
import matplotlib.pyplot as plt #pour la visualisation
import seaborn as sns #pour visualisation

from fonctions_utiles import * #importer les fonctions que j'ai écrites
from imblearn.over_sampling import SMOTE


##Outils pour les modèles

import xgboost as xgb
from sklearn.model_selection import train_test_split #séparation des bases en train et test
from sklearn import metrics
from sklearn.metrics import roc_curve, classification_report, r2_score, confusion_matrix, accuracy_score #pour les métriques de performance
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, cross_val_score #pour la validation croisée
from sklearn.model_selection import GridSearchCV #pour la recherche des meilleurs hyperparamètres
from yellowbrick.model_selection import ValidationCurve #visualisation de la validation
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV


##Modules pour les modèles
from sklearn.linear_model import LogisticRegression #modèle de régression logistique
from sklearn.linear_model import LinearRegression #modèle de régression linéaire
from sklearn.tree import DecisionTreeClassifier #modèle d'arbre de décision
from sklearn.ensemble import RandomForestClassifier #modèle de forêts aléatoires
from sklearn.neighbors import KNeighborsClassifier #modèle pour le KNN
from sklearn.ensemble import VotingClassifier #modèle pour vote des classfiers
from sklearn import tree #modèle pour arbre de décision avec possibilité de visualisation
from sklearn.preprocessing import PolynomialFeatures #modèle pour la régression polynomiale
import joblib #pour sauvegarder les modèles

##Ma base de données
data_soil = pd.read_excel('data_clean.xlsx')
##Les variables intéressantes
var_cibles_quant=['soft_wheat_area_km2', 'corn_grain_area_km2', 'barley_area_km2',
                'sunflower_area_km2', 'sugarbeet_area_km2']

var_cibles_qual=['grain_mais', 'ble_tendre', 'orge', 'tournsol', 'bettrave_a_sucre']

var_numeriques=['clay_0to30cm_percent', 'silt_0to30cm_percent', 'sand_0to30cm_percent', 'ph_h2o_0to30cm',
       'organic_carbon_0to30cm_percent', 'bdod_0to30cm',
       'cfvo_0to30cm_percent']

##Traitement des variables quantitatives
#boxplot(data_soil,var_numeriques)
correct_outliers(data_soil,var_numeriques)
#boxplot(data_soil,var_numeriques)



### Variable ble_tendre

#### Visualisation
visual_xqual_yqual("ble_tendre",var_cibles_qual,data_soil)
visual_xqual_yquant("ble_tendre",var_numeriques,data_soil)


variables_retenues_ble_tendre=var_numeriques+var_cibles_qual
data_ble_tendre=data_soil[variables_retenues_ble_tendre]
y_ble_tendre=data_ble_tendre['ble_tendre']
X_ble_tendre=data_ble_tendre.drop(['ble_tendre'], axis=1)

X_train_ble_tendre, X_test_ble_tendre, y_train_ble_tendre, y_test_ble_tendre = train_test_split(X_ble_tendre, y_ble_tendre, test_size=0.3, random_state=42)
"""
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_ble_tendre_resampled, y_train_ble_tendre_resampled = smote.fit_resample(X_train_ble_tendre, y_train_ble_tendre)

##Calibrage du modele RandomForest
mini=10
maxi=100
n_estimators = [int(x) for x in np.linspace(mini, maxi, 10)]
max_features = ['log2', 'sqrt']
max_depth = [10,20,30]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
criterion = ['gini', 'entropy']
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators, 
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion': criterion,
               'bootstrap': bootstrap
              }

rf_Model = RandomForestClassifier(random_state=42,class_weight="balanced")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = GridSearchCV(estimator=rf_Model, param_grid = params_grid, cv=cv)
#rf_grid.fit(X_train_ble_tendre_resampled, y_train_ble_tendre_resampled)

#print("Meilleurs hyperparametres",rf_grid.best_params_)
#print("score", rf_grid.best_score_)


model_ble_tendre = RandomForestClassifier(bootstrap=False,
 criterion='entropy',
 max_depth=10,
 max_features='log2',
 min_samples_leaf=2,
 min_samples_split=5,
 n_estimators=10,
 class_weight="balanced")
model_ble_tendre.fit(X_train_ble_tendre_resampled,y_train_ble_tendre_resampled)
print("Score sur le test",model_ble_tendre.score(X_test_ble_tendre,y_test_ble_tendre))

#metriques de performance
matrix_confusion(model_ble_tendre,X_test_ble_tendre, y_test_ble_tendre)
courbe_roc_AUC(model_ble_tendre,X_test_ble_tendre, y_test_ble_tendre)


joblib.dump(model_ble_tendre, "model_ble_tendre.pkl")

"""
#Variable ble_tendre mais numerique
variables_retenues_ble_tendre_num=var_numeriques+var_cibles_quant
data_ble_tendre_num=data_soil[variables_retenues_ble_tendre_num]
y_ble_tendre_num=data_ble_tendre_num["soft_wheat_area_km2"]
X_ble_tendre_num=data_ble_tendre_num.drop(['soft_wheat_area_km2'], axis=1)

X_train_ble_tendre_num, X_test_ble_tendre_num, y_train_ble_tendre_num, y_test_ble_tendre_num = train_test_split(X_ble_tendre_num, y_ble_tendre_num, test_size=0.3, random_state=42)


##Regression XGBoost
xgboost_mais_num= xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
xgboost_mais_num.fit(X_train_ble_tendre_num,y_train_ble_tendre_num)
y_pred_ble_tendre_num_xg=xgboost_mais_num.predict(X_test_ble_tendre_num)
rr=r2_score(y_test_ble_tendre_num, y_pred_ble_tendre_num_xg)
print("R2 avec xg",rr)
joblib.dump(xgboost_mais_num, "model_ble_tendre_num.pkl")


### Variable ble_tendre
### Variable orge
### Variable tournsol
### Variable bettrave à sucre


"""
Vote_Model = VotingClassifier([('SGD', model_1), 
                            ('Tree', model_2),
                            ('KNN', model_3)],
                          voting='hard')

for model in (model_1, model_2, model_3, Vote_Model):
    model.fit(X_train, y_train)
    print(model.__class__.__name__, model.score(X_test, y_test))



Model = tree.DecisionTreeClassifier()
Model = Model.fit(X_train, y_train)
tree.plot_tree(Model)






##KNN
Model = KNeighborsClassifier(n_neighbors=10)
Model.fit(X_train, Y_train)

print('Test score', Model.score(X_test, Y_test))


###REGRESSION LOGISTIC
model1 = LogisticRegression()
#scores
print("Accuracy Logit:",metrics.accuracy_score(y_test, y_pred_logit))
print("Precision Logit:",metrics.precision_score(y_test, y_pred_logit))
print("Recall Logit:",metrics.recall_score(y_test, y_pred_logit))
print("F1 Score Logit:",metrics.f1_score(y_test, y_pred_logit))



#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
reg=LinearRegression(fit_intercept=True)
model = reg.fit(train,train_label)
predict = model.predict(test)

print(r2_score(test_label,predict))

#POLYNOMIAL REGRESSION
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X_train)
"""





