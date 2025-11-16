# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import pyreadr




# %%
#Lecture des bases de données

# Lecture des fichier .rda
data_freq = pyreadr.read_r(r'C:\Users\kabir\OneDrive\Documents\master CNAM\Master 1\Python_cours\fremotor1freq0304a.rda')
data_prem = pyreadr.read_r(r'C:\Users\kabir\OneDrive\Documents\master CNAM\Master 1\Python_cours\fremotor1prem0304a.rda')

#*_format de data_freq et data_prem
print(type(data_freq))  

# Explorer le contenu
# Liste des objets contenus dans le fichier
print(data_freq.keys())
print(data_prem.keys())  

# Extraire le dataframe principal
df_freq = list(data_freq.values())[0]
print(df_freq.columns.tolist())       # Afficher les colonnes du dataframe
df_prem = list(data_prem.values())[0]
print(df_prem.columns.tolist())       # Afficher les colonnes du dataframe




#%%
#Préparation de la base de données complète

# Choix des colonnes du Data Frame des fréquences
df_freq = df_freq[['IDpol', 'Year', 'Damage']]  #Damage est notre variable cible (Y)

# Affichage des informations sur les Data Frames 
print("Les informations sur la base de données des fréquences est :",df_freq.info())
print("Les informations sur la base de données des primes est :", df_prem.info())

# Descriptions des données 
print("La description des données des fréquences est :", df_freq.describe(include='all'))
print("La description des données des primes :", df_prem.describe(include='all'))

# Remplacer les Nan par 'infomanquante'
df_prem[["JobCode", "MaritalStatus"]] = df_prem[["JobCode", "MaritalStatus"]].astype(str).fillna("infomanquante")
## Jointure des tables 
df_merge = pd.merge(df_prem, df_freq, on = ['IDpol', 'Year'], how = 'left')
#df = pd.dropna()
df_merge = df_merge[df_merge['PremDamAll'] > 0]                #Prime des Dommages (à modifier selon la variable cible sur laquelle on travaille)




#%%
# Choix des variables en provenance du Data Frame des primes 
col_df_prem = ['IDpol', 'Year', 'DrivAge', 'DrivGender', 'MaritalStatus', 'BonusMalus',  
                'LicenceNb', 'PayFreq', 'JobCode', 'VehAge', 'VehClass', 'VehPower', 
                'VehGas', 'VehUsage', 'Garage', 'Area', 'Region', 'Channel', 'Marketing', 
                'Damage']
df_merge = df_merge[col_df_prem]

print(" Données des 2 premières lignes :", df_merge.head(2))
print(" Les informations des données après jointure :", df_merge.info())
print(" La description des données après jointure :", df_merge.describe(include='all'))
print("Le type des variables :", df_merge.dtypes)
# df['Damage'] = df['Damage'].fillna(0)  # Remplacer les NaN par 0 dans la variable cible

# Identifier les variables numériques et catégorielles 
num_vars = df_merge.select_dtypes(include=['int', 'float', 'number']).columns.tolist()
cat_vars = df_merge.select_dtypes(include=['object', 'category']).columns.tolist()
print ("Les variables numériques sont :", num_vars)
print ("Les variables catégorielles sont :", cat_vars)




#%%
# Analyse exploratoire des données (EDA)

target = 'Damage'          #Variable cible


# Analyse univariée
  # Variables numériques
for col in num_vars:
    print(f"\n Variable numérique : {col}")  
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(df_merge[col], kde=True)
    plt.title(f"Distribution de {col}")

    plt.subplot(1,2,2)
    sns.boxplot(x=df_merge[col])
    plt.title(f"Boxplot de {col}")
    plt.show()


    # Variables catégorielles 
for col in cat_vars:
    print(f"\n Variable catégorielle : {col}")
    print(df_merge[col].value_counts(normalize=True) * 100)
    plt.figure(figsize=(8,4))
    sns.countplot(x=df_merge[col], order=df_merge[col].value_counts().index)
    plt.title(f"Répartition de {col}")
    plt.xticks(rotation=45)
    plt.show()



# Analyse bivariées
  # Variables numériques Vs Variable cible
for col in num_vars:
    if col != target:
       plt.figure(figsize=(6,4))
       sns.boxplot(x=target, y=col, data=df_merge)
       plt.title(f"{col} selon {target}")
       plt.show()
  

  # Variables catégorielles Vs Variable cible
for col in cat_vars:
    if col != target:
        print(f"\n Relation entre {col} et {target}")
        print(pd.crosstab(df_merge[col], df_merge[target], normalize='index') * 100)
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue=target, data=df_merge)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()



# Tests de dépendance
  # Test de Spearman pour les variables numériques
def spearman_test(df, target, num_vars):
    results = []
    for col in num_vars:
        if col == target:
            continue

        rho, pval = stats.spearmanr(df[col], df[target], nan_policy='omit')
        results.append([col, rho, pval])

    return pd.DataFrame(results, columns=["Variable", "Spearman_rho", "p_value"])

    # Application du test de Spearman
df_spearman = spearman_test(df_merge, target="Damage", num_vars=num_vars)
print("Test de Spearman \n", df_spearman.sort_values("p_value"))

# Spearman_rho mesure l’intensité d’une relation monotone    &&     p-value mesure si la corrélation est statistiquement significative
  # Si p-value < 0.05 ET |rho| > 0.1, on peut considérer qu'il existe une relation significative entre la variable et la cible.
  # Si p-value < 0.05 MAIS rho très faible (< 0.05), la relation est statistiquement significative mais pas forcément pertinente.
  # Si p-value > 0.05, pas de dépendance
    # Ici, on peut envisager de retirer les variables 'Year' et 'LicenceNb'

  # Test ANOVA pour les variables catégorielles
def anova_test(df, target, cat_vars):
    results = []
    for col in cat_vars:
        groups = [df[df[col] == m][target].dropna() for m in df[col].unique()]
        
        F, pval = stats.f_oneway(*groups)
        results.append([col, F, pval])

    return pd.DataFrame(results, columns=["Variable", "F_statistic", "p_value"])

    # Application du test ANOVA
df_anova = anova_test(df_merge, target="Damage", cat_vars=cat_vars)
print("Test ANOVA \n", df_anova.sort_values("p_value"))

# F_statistic : mesure la différence entre les moyennes des groupes   &&   p-value : teste si au moins une catégorie influence significativement le target
  # p < 0.05 : au moins une catégorie influence significativement le target, variable explicative utile
  # p ≥ 0.05 : aucune catégorie n'influence significativement le target, variable explicative peut être retirée du modèle
    # Ici, on peut envisager de retirer les variables : 'JobCode', 'Channel', 'DrivGender', 'VehClass', 'VehGas', ... 
      



#%%
# Binarisation des variables catégorielles
df = pd.get_dummies(df_merge, columns = [var for var in cat_vars if var != 'IDpol'], drop_first=True)
print(df.columns.tolist()) 




#%%
### Import des Bibliothèques\ Modules\ Classes'
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Création de l'exposure
df['exposure'] = 1
df['log_exposure'] = np.log(df['exposure'])

all_var = list(df.columns)
print(all_var)




#%%
### Prédiction de la fréquence des sinistres : 'Damage'  - Modèle avec toutes les variables 

# Variables explicatives (X) et réponse (y)
X = df[['Year', 'DrivAge', 'BonusMalus', 'LicenceNb', 'VehAge', 'DrivGender_M',
        'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single', 
        'MaritalStatus_Widowed', 'MaritalStatus_nan', 'PayFreq_Half-yearly', 'PayFreq_Monthly', 
        'PayFreq_Quarterly', 'JobCode_Farmer', 'JobCode_Other', 'JobCode_Private employee',
        'JobCode_Public employee', 'JobCode_Retailer', 'JobCode_Retiree', 'JobCode_nan', 'VehClass_Cheaper', 
        'VehClass_Cheapest', 'VehClass_Expensive', 'VehClass_Medium', 'VehClass_Medium high', 
        'VehClass_Medium low', 'VehClass_More expensive', 'VehClass_Most expensive',
        'VehPower_P11', 'VehPower_P12', 'VehPower_P13', 'VehPower_P14', 'VehPower_P15', 
        'VehPower_P16', 'VehPower_P17', 'VehPower_P2', 'VehPower_P4', 'VehPower_P5', 
        'VehPower_P6', 'VehPower_P7', 'VehPower_P8', 'VehPower_P9', 'VehGas_Regular', 
        'VehUsage_Professional', 'VehUsage_Professional run', 'Garage_Closed zbox', 
        'Garage_Opened collective parking', 'Garage_Street', 'Area_A12', 'Area_A2', 'Area_A3', 
        'Area_A4', 'Area_A5', 'Area_A6', 'Area_A7', 'Area_A8', 'Area_A9', 'Region_Headquarters', 
        'Region_Paris area', 'Region_South West', 'Channel_B', 'Channel_L', 'Marketing_M2',
        'Marketing_M3', 'Marketing_M4']]
y = df['Damage']
offset = df['log_exposure'] # Logarithme de l'exposure

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test, offset_train, offset_test = train_test_split(X, y, offset, test_size=0.25, random_state=42)

# Entrainement du modèle de régression de Poisson
model = PoissonRegressor(alpha=1e-12, max_iter=1000) # alpha contrôle la régularisation (L2)
model.fit(X_train, y_train, sample_weight=np.exp(offset_train)) 

##Pour étudier le modèle, on peut calculer différentes métriques.
 # Prédictions
y_pred = model.predict(X_test) * np.exp(offset_test) # Réintégration de l'offset dans les prédictions


# Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('Coefficients :', model.coef_)
print('Intercept :', model.intercept_)
print('Mean Squared Error :', mse)
print('Mean Absolute Error :', mae)


# Affichage des prédictions
predictions = pd.DataFrame({'Year': X_test['Year'], 'DrivAge' : X_test['DrivAge'], 'BonusMalus': X_test['BonusMalus'], 
                            'LicenceNb' : X_test['LicenceNb'], 'VehAge' : X_test['VehAge'], 'DrivGender_M' : X_test['DrivGender_M'],
                            'MaritalStatus_Divorced' : X_test['MaritalStatus_Divorced'], 'MaritalStatus_Married' : X_test['MaritalStatus_Married'], 
                            'MaritalStatus_Single' : X_test['MaritalStatus_Single'], 'MaritalStatus_Widowed' : X_test['MaritalStatus_Widowed'], 'MaritalStatus_nan' : X_test['MaritalStatus_nan'], 
                            'PayFreq_Half-yearly' : X_test['PayFreq_Half-yearly'], 'PayFreq_Monthly' : X_test['PayFreq_Monthly'], 
                            'PayFreq_Quarterly' : X_test['PayFreq_Quarterly'], 'JobCode_Farmer' : X_test['JobCode_Farmer'], 'JobCode_Other' : X_test['JobCode_Other'], 
                            'JobCode_Private employee' : X_test['JobCode_Private employee'],'JobCode_Public employee' : X_test['JobCode_Public employee'], 
                            'JobCode_Retailer' : X_test['JobCode_Retailer'], 'JobCode_Retiree' : X_test['JobCode_Retiree'], 'JobCode_nan' : X_test['JobCode_nan'], 'VehClass_Cheaper' : X_test['VehClass_Cheaper'], 
                            'VehClass_Cheapest' : X_test['VehClass_Cheapest'], 'VehClass_Expensive' : X_test['VehClass_Expensive'],
                            'VehClass_Medium' : X_test['VehClass_Medium'], 'VehClass_Medium high' : X_test['VehClass_Medium high'], 
                            'VehClass_Medium low' : X_test['VehClass_Medium low'], 'VehClass_More expensive' : X_test['VehClass_More expensive'], 
                            'VehClass_Most expensive' : X_test['VehClass_Most expensive'],'VehPower_P11' : X_test['VehPower_P11'], 'VehPower_P12' : X_test['VehPower_P12'], 
                            'VehPower_P13' : X_test['VehPower_P13'], 'VehPower_P14' : X_test['VehPower_P14'], 'VehPower_P15' : X_test['VehPower_P15'], 
                            'VehPower_P16' : X_test['VehPower_P16'], 'VehPower_P17' : X_test['VehPower_P17'], 'VehPower_P2' : X_test['VehPower_P2'], 
                            'VehPower_P4' : X_test['VehPower_P4'], 'VehPower_P5' : X_test['VehPower_P5'], 'VehPower_P6' : X_test['VehPower_P6'], 
                            'VehPower_P7' : X_test['VehPower_P7'], 'VehPower_P8' : X_test['VehPower_P8'], 'VehPower_P9' : X_test['VehPower_P9'], 
                            'VehGas_Regular' : X_test['VehGas_Regular'],'VehUsage_Professional' : X_test['VehUsage_Professional'], 
                            'VehUsage_Professional run' : X_test['VehUsage_Professional run'], 'Garage_Closed zbox' : X_test['Garage_Closed zbox'], 
                            'Garage_Opened collective parking' : X_test['Garage_Opened collective parking'], 'Garage_Street' : X_test['Garage_Street'], 
                            'Area_A12' : X_test['Area_A12'], 'Area_A2' : X_test['Area_A2'], 'Area_A3': X_test['Area_A3'], 
                            'Area_A4' : X_test['Area_A4'], 'Area_A5': X_test['Area_A5'], 'Area_A6': X_test['Area_A6'], 'Area_A7': X_test['Area_A7'], 
                            'Area_A8': X_test['Area_A8'], 'Area_A9': X_test['Area_A9'], 'Region_Headquarters': X_test['Region_Headquarters'], 
                            'Region_Paris area': X_test['Region_Paris area'], 'Region_South West': X_test['Region_South West'], 'Channel_B': X_test['Channel_B'], 
                            'Channel_L': X_test['Channel_L'], 'Marketing_M2': X_test['Marketing_M2'],'Marketing_M3': X_test['Marketing_M3'], 
                            'Marketing_M4': X_test['Marketing_M4']})

print(predictions)





#%% 
### Courbe de Lorenz

def lorenz_curve(data,data_predict):
     
   data_df = pd.DataFrame(data,columns=['Damage']).reset_index()
   data_predict_df = pd.DataFrame(data_predict,columns=['y_pred']).reset_index()
   data_df = pd.concat((data_df,data_predict_df), axis = 'columns')
   display(data_df)
   # Trier y_test en utilisant les indices triés
   sorted_data = data_df.sort_values(by='y_pred', ascending=True, inplace=False)
   #sorted_data = np.sort(data) # Trier les données dans l'ordre croissant
   data_df['cumulative_income'] = data_df['Damage'].cumsum() # Somme cumulée des données
   data_df['cumulative_income'] = data_df['cumulative_income'] / data_df['cumulative_income'].iloc[-1] # iloc[-1] permet de normaliser à 1
   data_df['cumulative_population'] = np.linspace(0, 1, len(data_df)) # Population cumulée normalisée
   return data_df
 
# Calcul de la courbe de Lorenz
data_df = lorenz_curve(y_test,y_pred)

 # Tracé de la courbe
plt.figure(figsize=(8, 8))
plt.plot(data_df['cumulative_population'], data_df['cumulative_income'], label='Courbe de Lorenz', color='red', linewidth=2)
plt.plot([0, 1], [0, 1], label='Égalité parfaite', color='blue', linestyle='--') # Ligne d'égalité parfaite
plt.title('Courbe de Lorenz')
plt.xlabel('% de la population cumulée')
plt.ylabel('% cumulé de la variable à expliquer')
plt.legend()
plt.grid(True)
plt.show()

# Le modèle est presque égale à 50% de la moyenne. 
# Cela signifie que le modèle n'a pas une grande capacité prédictive.
  
  # Suggestions d'amélioration :
# Régulatrisation L1 (Lasso) et L2 (Ridge) peuvent être explorées pour améliorer la performance du modèle.
# L'ajustement des hyperparamètres, comme le paramètre de régularisation alpha, peut également aider à optimiser le modèle.
# Réduction de dimensionnalité avec ACP


# Dans les lignes qui suivent je vais utiliser Ridge, Lasso et Stepwise (comme dans le cours) pour améliorer le modèle. 





#%% 
### Stepwise

from sklearn.metrics import mean_poisson_deviance


# Initialisation des variables
remaining = list(X_train.columns)
selected = []
best_score = np.inf    # Pour la Poisson deviance : plus petit = meilleur

# Fonction d'évaluation du modèle
def evaluate_model(vars_list):
    model = PoissonRegressor(alpha=1e-12, max_iter=2000)
    model.fit(X_train[vars_list], y_train, sample_weight=np.exp(offset_train))
    y_pred = model.predict(X_test[vars_list])
    return mean_poisson_deviance(y_test, y_pred)


# Forward Stepwise Selection-
while remaining:
    scores_with_candidates = []
    
    for candidate in remaining:
        candidate_vars = selected + [candidate]
        score = evaluate_model(candidate_vars)
        scores_with_candidates.append((score, candidate))
    
    # Choix de la variable qui réduit le plus la deviance
    scores_with_candidates.sort()
    best_new_score, best_candidate = scores_with_candidates[0]

    if best_new_score < best_score:
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_new_score
        print(f"Ajoutée : {best_candidate}, Nouvelle deviance : {best_new_score:.4f}")
    else:
        break

print("\nVariables sélectionnées :", selected)

# Réentraînement final

model_stepwise = PoissonRegressor(alpha=1e-12, max_iter=2000)
model_stepwise.fit(X_train[selected], y_train, sample_weight=np.exp(offset_train))

y_pred_stepwise = model_stepwise.predict(X_test[selected])
deviance_finale = mean_poisson_deviance(y_test, y_pred_stepwise)

print(f"\nDeviance finale avec les variables sélectionnées : {deviance_finale:.4f}")





#%%
# Application du modèle avec Stepwise 
var_stepwise = ['BonusMalus', 'Region_Headquarters', 'Channel_B', 'PayFreq_Monthly', 'MaritalStatus_Married', 
                   'LicenceNb', 'VehClass_Expensive', 'VehPower_P14', 'VehPower_P11', 'VehUsage_Professional run', 
                   'VehPower_P9', 'Area_A2', 'MaritalStatus_Widowed', 'VehPower_P2']

X_stepwise_train = X_train[var_stepwise]
X_stepwise_test  = X_test[var_stepwise]
y_stepwise_train = y_train
y_stepwise_test  = y_test
offset_stepwise_train = offset_train
offset_stepwise_test = offset_test

# Entrainement du modèle de régression de Poisson
model.fit(X_stepwise_train, y_stepwise_train, sample_weight=np.exp(offset_stepwise_train)) 

##Pour étudier le modèle, on peut calculer différentes métriques.
 # Prédictions
y_stepwise_pred = model.predict(X_stepwise_test) * np.exp(offset_stepwise_test) 

# Évaluation des performances
mse_s = mean_squared_error(y_stepwise_test, y_stepwise_pred)
mae_s = mean_absolute_error(y_stepwise_test, y_stepwise_pred)
print('Coefficients :', model.coef_)
print('Intercept :', model.intercept_)
print('Mean Squared Error :', mse_s)
print('Mean Absolute Error :', mae_s)
 
data_df_s = lorenz_curve(y_stepwise_test,y_stepwise_pred)       





#%%
### Ridge

ridge_model = PoissonRegressor(alpha=0.1, max_iter = 1000)  
    
# Ajuster le modèle
ridge_model.fit(X_train, y_train, sample_weight=np.exp(offset_train))

# Prédire sur les données de test
y_pred_ridge = ridge_model.predict(X_test)

# Afficher les coefficients du modèle
coefficients_ridge = pd.DataFrame({'Variable': X.columns, 
                                   'Coefficient': ridge_model.coef_})

display(coefficients_ridge)

coefficients_ridge['AbsCoef'] = coefficients_ridge['Coefficient'].abs()
coefficients_ridge = coefficients_ridge.sort_values('AbsCoef', ascending=False)

display(coefficients_ridge)





#%%
# Application du modèle avec Ridge
vars_ridge = coefficients_ridge.loc[coefficients_ridge['AbsCoef'] > 1e-4, 'Variable'].tolist()
print("Les variables à retenir pour après la régularisation avec Ridge sont \n", vars_ridge)

X_ridge_train = X_train[vars_ridge]
X_ridge_test  = X_test[vars_ridge]
y_ridge_train = y_train
y_ridge_test  = y_test
offset_ridge_train = offset_train
offset_ridge_test = offset_test

# Entrainement du modèle de régression de Poisson
model.fit(X_ridge_train, y_ridge_train, sample_weight=np.exp(offset_ridge_train)) 

##Pour étudier le modèle, on peut calculer différentes métriques.
 # Prédictions
y_ridge_pred = model.predict(X_ridge_test) * np.exp(offset_ridge_test) 

# Évaluation des performances
mse_r = mean_squared_error(y_ridge_test, y_ridge_pred)
mae_r = mean_absolute_error(y_ridge_test, y_ridge_pred)
print('Coefficients :', model.coef_)
print('Intercept :', model.intercept_)
print('Mean Squared Error :', mse_r)
print('Mean Absolute Error :', mae_r)
 
data_df_r = lorenz_curve(y_ridge_test,y_ridge_pred)       



#%%
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson


#%%
print(X_train.dtypes)
print(X_train.select_dtypes(include=["object"]).columns)
#%%
### Lasso

# Ajout d'une constante pour l'intercept
#X_train_constant = sm.add_constant(X_train)
#X_test_constant  = sm.add_constant(X_test)
X_train_constant = X_train
X_test_constant  = X_test

# Convertir les booléens en entiers - statsmodels ne gère pas les booléens
X_train_constant = X_train_constant.astype({col: "int" for col in X_train_constant.columns if X_train_constant[col].dtype == bool})
X_test_constant  = X_test_constant.astype({col: "int" for col in X_test_constant.columns  if X_test_constant[col].dtype == bool})

# Instancier le GLM Poisson
model_lasso = sm.GLM(
    y_train,
    X_train_constant,
    family=sm.families.Poisson(),
    offset=offset_train
)

# Fit LASSO
resultats_lasso = model_lasso.fit_regularized(alpha=0.1, L1_wt=1.0)

# Extraction des coefficients
coef_l = resultats_lasso.params
coefficients_lasso = pd.DataFrame({
    "Variable": X_train_constant.columns,
    "Coefficient": coef_l
})

display(coefficients_lasso)

# Filtrer les variables sélectionnées
selected_variables_l = coefficients_lasso[coefficients_lasso["Coefficient"] != 0]

display("Variables sélectionnées par LASSO :")
display(selected_variables_l)
display(selected_variables_l.shape)




#%%
# Application du modèle avec Lasso 
var_lasso = ['Year']

X_lasso_train = X_train[var_lasso]
X_lasso_test  = X_test[var_lasso]
y_lasso_train = y_train
y_lasso_test  = y_test
offset_lasso_train = offset_train
offset_lasso_test = offset_test

# Entrainement du modèle de régression de Poisson
model.fit(X_lasso_train, y_lasso_train, sample_weight=np.exp(offset_lasso_train)) 

##Pour étudier le modèle, on peut calculer différentes métriques.
 # Prédictions
y_lasso_pred = model.predict(X_lasso_test) * np.exp(offset_lasso_test) 

# Évaluation des performances
mse_l = mean_squared_error(y_lasso_test, y_lasso_pred)
mae_l = mean_absolute_error(y_lasso_test, y_lasso_pred)
print('Coefficients :', model.coef_)
print('Intercept :', model.intercept_)
print('Mean Squared Error :', mse_l)
print('Mean Absolute Error :', mae_l)
 
data_df_l = lorenz_curve(y_lasso_test,y_lasso_pred)       





#%%
### préparation des variables conseillées par les analyse de dépendance

var_dependance = ['DrivAge', 'BonusMalus', 'PayFreq', 'VehAge','VehUsage', 'Garage', 'Damage'] 

df_dep = df_merge[var_dependance]

# Binarisation des variables catégorielles
df_dep = pd.get_dummies(df_dep,
                        columns=[col for col in var_dependance if df_dep[col].dtype in ['object', 'category']],
                        drop_first=True
)
print(df_dep.columns.tolist()) 

# Variables explicatives X_dep et réponse y_dep
X_dep = df[df_dep.columns[df_dep.columns != "Damage"].tolist()]
y_dep = df_dep['Damage']

#Exposure
df_dep['exposure'] = 1
df_dep['log_exposure'] = np.log(df_dep['exposure'])
offset_dep = df_dep['log_exposure'] 


#%% 
### Application du modèle avec les variables issues des analyses de dépendance

# Division des données en ensembles d'entraînement et de test
X_dep_train, X_dep_test, y_dep_train, y_dep_test, offset_dep_train, offset_dep_test = train_test_split(X_dep, y_dep, offset_dep, test_size=0.25, random_state=42)

# Fit du modèle de régression de Poisson
model.fit(X_dep_train, y_dep_train, sample_weight=np.exp(offset_dep_train)) 

##Pour étudier le modèle, on peut calculer différentes métriques.
 # Prédictions
y_dep_pred = model.predict(X_dep_test) * np.exp(offset_dep_test) # Réintégration de l'offset dans les prédictions


# Évaluation des performances
mse_d = mean_squared_error(y_dep_test, y_dep_pred)
mae_d = mean_absolute_error(y_dep_test, y_dep_pred)
print('Coefficients :', model.coef_)
print('Intercept :', model.intercept_)
print('Mean Squared Error :', mse_d)
print('Mean Absolute Error :', mae_d)

data_df_dep = lorenz_curve(y_dep_test,y_dep_pred)


#%%
### Comparaison des courbes de Lorenz entre les modèles 

plt.figure(figsize=(8, 8))
plt.plot(data_df['cumulative_population'], data_df['cumulative_income'], label='Modèle basique', color='red', linewidth=2)
plt.plot(data_df_s['cumulative_population'], data_df_s['cumulative_income'], label='Régularisation Stepwise', color='green', linewidth=2, linestyle = '--')
plt.plot(data_df_r['cumulative_population'], data_df_r['cumulative_income'], label='Régularisation Ridge', color='purple', linewidth=2, linestyle = '--')
plt.plot(data_df_l['cumulative_population'], data_df_l['cumulative_income'], label='Régularisation Lasso', color='black', linewidth=2, linestyle = '--')
plt.plot(data_df_dep['cumulative_population'], data_df_dep['cumulative_income'], label='Modèle après Tests', color='orange', linewidth=2, linestyle = '--')
plt.plot([0, 1], [0, 1], label='Égalité parfaite', color='blue', linestyle='--') # Ligne d'égalité parfaite
plt.title('Courbe de Lorenz')
plt.xlabel('% de la population cumulée')
plt.ylabel('% cumulé de la variable à expliquer')
plt.legend()
plt.grid(True)
plt.show()




#%% Observations et suite des travaux
#Nous constatons que même après la régularisation et la sélection de variables, les performances du modèle restent relativement similaires.
# En regardant les mse, nous constatons que les différences sont minimes entre les différents modèles.
# Cela suggère que les variables initiales sélectionnées ont une influence limitée sur la capacité prédictive du modèle
# La solution pourrait résider dans l'exploration de nouvelles variables explicatives ou dans l'utilisation de techniques de modélisation plus avancées comme les modèles d'ensemble (Random Forest, Gradient Boosting) ou les réseaux de neurones.
# Une autre piste serait d'explorer des transformations non linéaires des variables explicatives ou d'interactions entre elles comme suggéré dans le cours sur la réduction de dimensionnalité.