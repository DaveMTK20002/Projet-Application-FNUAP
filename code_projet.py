## Implémentation des différents modèles

##Modules de traitement
import numpy as np #pour les vecteurs, array, uni-dimensionnel
import pandas as pd #pour les dataframes
import matplotlib.pyplot as plt #pour la visualisation
import seaborn as sns #pour visualisation

from fonctions_utiles import * #importer les fonctions que j'ai écrites



##Outils pour les modèles

from sklearn.model_selection import train_test_split #séparation des bases en train et test
from sklearn import metrics
from sklearn.metrics import roc_curve, classification_report, r2_score, confusion_matrix, accuracy_score #pour les métriques de performance
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, cross_val_score #pour la validation croisée
from sklearn.model_selection import GridSearchCV #pour la recherche des meilleurs hyperparamètres
from yellowbrick.model_selection import ValidationCurve #visualisation de la validation

##Modules pour les modèles
from sklearn.linear_model import SGDClassifier #classificatin par gradient stochastique
from sklearn.linear_model import LogisticRegression #modèle de régression logistique
from sklearn.linear_model import LinearRegression #modèle de régression linéaire
from sklearn.tree import DecisionTreeClassifier #modèle d'arbre de décision
from sklearn.ensemble import RandomForestClassifier #modèle de forêts aléatoires
from sklearn.neighbors import KNeighborsClassifier #modèle pour le KNN
from sklearn.ensemble import VotingClassifier #modèle pour vote des classfiers
from sklearn import tree #modèle pour arbre de décision avec possibilité de visualisation
from sklearn.preprocessing import PolynomialFeatures #modèle pour la régression polynomiale

##Ma base de données
data_soil = pd.read_excel('data_clean.xlsx')
##Les variables intéressantes
var_cibles_quant=['soft_wheat_area_km2', 'hard_wheat_area_km2',
       'corn_grain_area_km2', 'corn_silage_area_km2', 'barley_area_km2',
       'rapeseed_area_km2', 'sunflower_area_km2', 'sugarbeet_area_km2',
       'vineyards_area_km2']

var_cibles_qual=['ble_tendre', 'ble_dur', 'grain_mais',
       'ensilage_mais', 'orge', 'colza', 'tournsol', 'bettrave_a_sucre',
       'vignobles']

var_numeriques=['clay_0to30cm_percent', 'silt_0to30cm_percent', 'sand_0to30cm_percent', 'ph_h2o_0to30cm',
       'organic_carbon_0to30cm_percent', 'bdod_0to30cm',
       'cfvo_0to30cm_percent']

##Traitement des variables quantitatives
boxplot(data_soil,var_numeriques)
correct_outliers(data_soil,var_numeriques)
boxplot(data_soil,var_numeriques)



### Variable grain_mais

#### Visualisation
visual_xqual_yqual("grain_mais",var_cibles_qual,data_soil)
visual_xqual_yquant("grain_mais",var_numeriques,data_soil)

variables_retenues_grain_mais=var_numeriques+var_cibles_qual
data_grain_mais=data_soil[variables_retenues_grain_mais]
print(variables_retenues_grain_mais)
y_grain_mais=data_grain_mais['grain_mais']
X_grain_mais=data_grain_mais.drop(['grain_mais'], axis=1)

X_train_grain_mais, X_test_grain_mais, y_train_grain_mais, y_test_grain_mais = train_test_split(X_grain_mais, y_grain_mais, test_size=0.3, random_state=0)

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

rf_Model = RandomForestClassifier(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = GridSearchCV(estimator=rf_Model, param_grid = params_grid, cv=cv)
#rf_grid.fit(X_train_grain_mais, y_train_grain_mais)

#print("Meilleurs hyperparametres",rf_grid.best_params_)
#print("score", rf_grid.best_score_)
model_grain_mais = RandomForestClassifier(bootstrap=False,
 criterion='gini',
 max_depth=10,
 max_features='log2',
 min_samples_leaf=2,
 min_samples_split=2,
 n_estimators=20)
model_grain_mais.fit(X_train_grain_mais,y_train_grain_mais)
print("Score sur le test",model_grain_mais.score(X_test_grain_mais,y_test_grain_mais))

#metriques de performance
matrix_confusion(model_grain_mais,X_test_grain_mais, y_test_grain_mais)
courbe_roc_AUC(model_grain_mais,X_test_grain_mais, y_test_grain_mais)
### Variable ble_tendre
### Variable ble_dur
### Variable ble_tendre

## Variable ensilage_mais

'''


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



#KFOLDS
cval = KFold(7)
cval = LeaveOneOut()
cval = ShuffleSplit(5, test_size=0.25)
cval=StratifiedKFold()
cross_val_score(tree.DecisionTreeClassifier(), X_train, y_train, cv=cval)

#ROC ET AUC
#AUC Calculation
Modl = tree.DecisionTreeClassifier().fit(X_train, y_train)





##KNN
Model = KNeighborsClassifier(n_neighbors=10)
Model.fit(X_train, Y_train)

print('Test score', Model.score(X_test, Y_test))

val_score=[]
for k in range (1, 50):#On fait varier l'hyperparamètre n_neighbors de 1 à 49
    score=cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, Y_train, cv=5, scoring='accuracy').mean()
    val_score.append(score)
plt.plot(val_score)


viz = ValidationCurve(
    KNeighborsClassifier(), param_name="n_neighbors",
    param_range=np.arange(1,50), cv=10, scoring="r2"
)

# Fit and show the visualizer
viz.fit(X_train, Y_train)
viz.show()

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
'''












