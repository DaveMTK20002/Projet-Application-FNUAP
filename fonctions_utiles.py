import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, classification_report, r2_score, confusion_matrix, accuracy_score #pour les métriques de performance

##ECRITURE DES FONCTIONS UTILES
def desc_var_qual(x,data):
    fig , ax = plt.subplots(figsize = (6, 4))
    sns.countplot(x = x, data = data)
    plt.title('Distribution de '+x)
    plt.show()
    print(data[x].value_counts())
    effectif_x_0 = len(data[data[x]==0])
    effectif_x_1 = len(data[data[x]==1])
    n = len(data)
    print("% favorable: ", round(effectif_x_1*100/n,2))
    print("% Non-favorable : ", round(effectif_x_0*100/n,2))
    print("Total:",n)

# Percentile Objects

# 10th Percentile
def q10(x):
    return x.quantile(0.1)

# 20th Percentile
def q20(x):
    return x.quantile(0.2)

# 30th Percentile
def q30(x):
    return x.quantile(0.3)

# 40th Percentile
def q40(x):
    return x.quantile(0.4)

# 50th Percentile
def q50(x):
    return x.quantile(0.5)

# 60th Percentile
def q60(x):
    return x.quantile(0.6)

# 70th Percentile
def q70(x):
    return x.quantile(0.7)

# 80th Percentile
def q80(x):
    return x.quantile(0.8)

# 90th Percentile
def q90(x):
    return x.quantile(0.9)

# 95th Percentile
def q95(x):
    return x.quantile(0.95)

# 99th Percentile
def q99(x):
    return x.quantile(0.99)

#resumer la distribution de x selon y
def distrib (x,y,data):
    return data.groupby([y]).agg({x: [q10, q30, q50, q70, q90, 'mean']})

def visual_xqual_yquant(x,liste_var_quant,data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(liste_var_quant):
        ax = plt.subplot((len(liste_var_quant)//3)+1, 3, i+1)
        sns.boxplot(x=x,y=col, data=data,ax=ax)
        ax.set_title(f"Lien entre {x} et {col}")
    plt.tight_layout()
    plt.show()


def visual_xqual_yqual(x,liste_var_qual,data):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(liste_var_qual):
        ax = plt.subplot((len(liste_var_qual)//3)+1, 3, i+1)
        sns.countplot(x=x, data=data, hue=col,ax=ax)
        ax.set_title(f"Lien entre {x} et {col}")
    plt.tight_layout()
    plt.show()

def visual_xquant_(numerical,data):
    sns.pairplot(data_soil[numerical])
    plt.show()

# Fonction de création des boxplots par variable et détermination des outliers
def boxplot(df,numerical):
    plt.figure()
    for i, col in enumerate(numerical):
        ax = plt.subplot((len(numerical)//3)+1, 3, i+1)
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()



## Fonction de correction des valeurs aberrantes

def correct_outliers(df,numerical):
    for var in df[numerical].columns:
        IQR = df[var].quantile(0.75) - df[var].quantile(0.25)
        lower = df[var].quantile(0.25) - (1.5*IQR)
        upper = df[var].quantile(0.75) + (1.5*IQR)
        df[var] = np.where(df[var]>upper,upper, np.where(df[var]<lower,lower,df[var]))
    print("Done !")


# ROC Curve et AUC
def courbe_roc_AUC(modele,X_test, y_test):
    y_pred_proba = modele.predict_proba(X_test)[::,1]
    nom_modele = type(modele).__name__
    AUC = metrics.roc_auc_score(y_test, y_pred_proba)
    fp, tp, _ = metrics.roc_curve(y_test, y_pred_proba)
    plt.plot(fp, tp, label = "{}, AUC = {:.3f}".format(nom_modele,AUC))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('{} ROC'.format(nom_modele))
    plt.legend(loc=4)
    plt.show()



#Confusion matrix
def matrix_confusion(modele,X_test, y_test):
    y_pred=modele.predict(X_test)
    matrix_conf= metrics.confusion_matrix(y_test, y_pred)
    cm_modele = pd.DataFrame(matrix_conf, index=['non-favorable', 'favorable'], columns=['non-favorable', 'favorable'])

    sns.heatmap(cm_modele, annot=True, cbar=None, cmap="Blues", fmt = 'g')
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()
    print(classification_report(y_test, y_pred))


def selection_var(y,liste):
    pass

if __name__ == "__main__":
##On teste les fonctions
    data_soil = pd.read_excel('data_clean.xlsx')
    print(desc_var_qual("orge",data_soil))
    print(distrib("silt_0to30cm_percent","orge",data_soil))
    print(visual_xqual_yquant("tournsol","silt_0to30cm_percent",data_soil))
    print(visual_xqual_yqual("ble_tendre","orge",data_soil))
