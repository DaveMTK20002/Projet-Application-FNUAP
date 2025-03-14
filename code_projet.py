## Implémentation des différents modèles

##Modules de traitement
import numpy as np #pour les vecteurs, array, uni-dimensionnel
import pandas as pd #pour les dataframes
import matplotlib.pyplot as plt #pour la visualisation
import seaborn as sns #pour visualisation

from fonctions_utiles import * #importer les fonctions que j'ai écrites
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

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



### Variable bettrave_a_sucre

#### Visualisation
visual_xqual_yqual("bettrave_a_sucre",var_cibles_qual,data_soil)
visual_xqual_yquant("bettrave_a_sucre",var_numeriques,data_soil)


variables_retenues_bettrave_a_sucre=var_numeriques+var_cibles_qual
data_bettrave_a_sucre=data_soil[variables_retenues_bettrave_a_sucre]
y_bettrave_a_sucre=data_bettrave_a_sucre['bettrave_a_sucre']
X_bettrave_a_sucre=data_bettrave_a_sucre.drop(['bettrave_a_sucre'], axis=1)

X_train_bettrave_a_sucre, X_test_bettrave_a_sucre, y_train_bettrave_a_sucre, y_test_bettrave_a_sucre = train_test_split(X_bettrave_a_sucre, y_bettrave_a_sucre, test_size=0.3, random_state=42)
"""
count_class_0 = y_train_bettrave_a_sucre.value_counts().min()  # Nombre d'exemples de la classe 0
count_class_1 = y_train_bettrave_a_sucre.value_counts().max()  # Nombre d'exemples de la classe 1
scale_pos_weight = count_class_1/count_class_0  # Calcul du scale_pos_weight
# Initialiser le modèle XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight  # Appliquer le poids de classe
)


param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}


random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)

# Effectuer la recherche sur les paramètres
random_search.fit(X_train_bettrave_a_sucre, y_train_bettrave_a_sucre)

print("Meilleurs hyperparametres",random_search.best_params_)
print("score", random_search.best_score_)


model_bettrave_a_sucre = random_search
print("Score sur le test",model_bettrave_a_sucre.score(X_test_bettrave_a_sucre,y_test_bettrave_a_sucre))

#metriques de performance
matrix_confusion(model_bettrave_a_sucre,X_test_bettrave_a_sucre, y_test_bettrave_a_sucre)
courbe_roc_AUC(model_bettrave_a_sucre,X_test_bettrave_a_sucre, y_test_bettrave_a_sucre)


joblib.dump(model_bettrave_a_sucre, "model_bettrave_a_sucre.pkl")
"""
#Variable bettrave_a_sucre mais numerique
variables_retenues_bettrave_a_sucre_num=var_numeriques+var_cibles_quant
data_bettrave_a_sucre_num=data_soil[variables_retenues_bettrave_a_sucre_num]
y_bettrave_a_sucre_num=data_bettrave_a_sucre_num["sugarbeet_area_km2"]
X_bettrave_a_sucre_num=data_bettrave_a_sucre_num.drop(['sugarbeet_area_km2'], axis=1)

X_train_bettrave_a_sucre_num, X_test_bettrave_a_sucre_num, y_train_bettrave_a_sucre_num, y_test_bettrave_a_sucre_num = train_test_split(X_bettrave_a_sucre_num, y_bettrave_a_sucre_num, test_size=0.3, random_state=42)


##Regression XGBoost
xgboost_mais_num= xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
xgboost_mais_num.fit(X_train_bettrave_a_sucre_num,y_train_bettrave_a_sucre_num)
y_pred_bettrave_a_sucre_num_xg=xgboost_mais_num.predict(X_test_bettrave_a_sucre_num)
rr=r2_score(y_test_bettrave_a_sucre_num, y_pred_bettrave_a_sucre_num_xg)
print("R2 avec xg",rr)
joblib.dump(xgboost_mais_num, "model_bettrave_a_sucre_num.pkl")
print(X_test_bettrave_a_sucre_num.columns)

### Variable bettrave_a_sucre
### Variable bettrave_a_sucre
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


