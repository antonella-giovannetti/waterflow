# Projet waterflow

## Veille

### Qu'est ce que Machine Learning Operations (MLOps)
Le MLOps est un ensemble de pratiques qui aide les data scientists et les ingénieurs à gérer plus efficacement le cycle de vie du machine learning (ML).

ll vise à combler le fossé entre le développement et l'exploitation du machine learning. L'objectif du MLOps est de s'assurer que les modèles de ML sont développés, testés et déployés de manière cohérente et fiable.

Le MLOps est de plus en plus important, car de plus en plus d'entreprises s'en servent pour prendre des décisions commerciales essentielles.

Il implique des tâches telles que :

- Suivi des tests : suivi des tests et des résultats pour identifier les meilleurs modèles
- Déploiement de modèles : déployer des modèles en production et les rendre accessibles aux applications
- Surveillance des modèles : la surveillance des modèles permet de détecter les problèmes ou les dégradations de performances
- Réentraînement des modèles : réentraîner des modèles avec de nouvelles données pour améliorer leurs performances

Le MLOps est essentiel pour s'assurer que les modèles de machine learning sont fiables, évolutifs et faciles à gérer dans les environnements de production.

src : https://cloud.google.com/discover/what-is-mlops?hl=fr

### Qu'est ce que MLFlow ?
MLflow est une plateforme open source qui permet de gérer le cycle de vie des modèles de Machine Learning. En particulier, grâce à MLflow, les modèles qui ont été entraînés à une date spécifique ainsi que les hyper-paramètres associés pourront être stockés, monitorés et réutilisés de manière efficace.

Un des principaux avantages est l'intégration avec des frameworks Python existants comme Scikit-Learn, XGBoost ou encore TensorFlow, rendant l'outil utilisable dans de nombreuses situations. De plus, son API Python est simple à utiliser, permettant de l'intégrer rapidement dans des projets existants.

La réalisation d’un projet de Machine Learning (ML) est compliquée pour les équipes en science des données. Elles sont confrontées à plusieurs défis.

Tout d’abord, il y a de nombreux outils et versions différents utilisés depuis la préparation des données jusqu’à l’entraînement des modèles. Cela peut causer des problèmes de compatibilité et de gestion.

Ensuite, il est difficile de suivre et de comprendre comment les modèles de ML sont entraînés en raison de la multitude de paramètres possibles. Que vous travailliez seul ou en équipe, savoir exactement comment chaque modèle a été entraîné devient un casse-tête.

De plus, il est souvent difficile de réutiliser le code sans des instructions détaillées. Que vous partagiez votre code avec un développeur pour une utilisation en production ou que vous deviez revenir sur un projet antérieur pour le débugger, il est essentiel de pouvoir reproduire les étapes du processus de ML.

Enfin, mettre un modèle en production peut être compliqué en raison des nombreux outils de déploiement et des environnements variés requis, ainsi que le développement en continue du modèle avec l’intégration de nouvelles données. Cela ajoute une couche de complexité supplémentaire au processus global.

src : http://datascientest.com/mlflow-tout-savoir

## Contexte du projet
Pour le projet Waterflow, il nous est demmandé de déveopper un outil décisionnel permettant l'identification d'eau potable à la consommation humaine.
Pour ce, nous devrions suivre des étapes de prétraitement des données et analyse exploratoire.Ce projet de Machine Learning vise à prédire si l'eau est potable ou non à partir de plusieurs paramètres physico-chimiques à l'aide d'un modèle Random Forest ou d'autres algorithmes. L'ensemble du pipeline de modélisation est suivi et enregistré avec **MLflow**.

## Contenu du projet

- `water_potability.csv` — Dataset contenant les échantillons d'eau
- `main.py` — Script principal d'entraînement et d'évaluation
- `README.md` — Ce fichier

## Analyse des données
Le dataset contient 9 mesures de la qualité de l'eau pour 3276 étendues d'eau différentes : 
- ph -> mesure l'acidité ou l'alcalinité de l'eau
- Hardness -> due principalement à la présence de calcium et magnésium
- Solids -> mesure la quantité totale de minéraux dissous
- Chloramines -> sous-produits de la chloration avec de l'ammoniaque 
- Sulfate -> présents naturellement dans les eaux souterraines et minérales 
- Conductivity -> indique la capacité de l'eau à conduire un courant électrique 
- Organic_carbon -> quantité de carbonne issu de composés organiques dans l'eau 
- Trihalomethanes -> composés formés lors de la chloration de l'eau contenant de la matière organique
- Turbidity -> mesure des particules en suspension qui diffusent la lumière
- Potability -> cible : indicateur binaire : 1, 0 = non potable

## Objectif

Prédire la **potabilité** de l’eau (`Potability = 0` ou `1`) en fonction des colonnes suivantes :

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity

## Installation

### Prérequis :
- Python 3.8+
- pip

### Installation des dépendances :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlflow
```


### Lancer MLflow localement :
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```
## Algo
Pour l'algoritheme, nous avions décider de faire un MLP, mais n'ayant pas réusii à l'intégrer dans le MLflow, nous nous sommes rabattues sur le Random Forest. Voici les manipulations effectuées:
n_estimators = 100

Séparation : 80% train / 20% test

Imputation des données manquantes par médiane

Normalisation via StandardScaler

### Résultats

On a obtenu les résultats suivants:
- la courbe ROC montre une tendance descendante régulière : cela montre que le modèle apprend bien et que les erreurs diminuent au fil des cycles. C’est signe d’un entraînement stable.la courbe fluctue un peu mais reste globalement cohérente, c’est normal et peut être lié à la régularisation ou au bruit dans les données.

- La matrice de confusion révèle que le modèle est assez performant pour détecter l’eau non potable, avec 355 prédictions correctes. Cependant, il présente une tendance à sous-classer l’eau potable, avec 179 cas d’eau saine incorrectement identifiés comme non potables. Ce déséquilibre peut refléter une sensibilité élevée mais une précision moindre du modèle sur la classe potable. Pour améliorer la fiabilité, il serait pertinent d’ajuster le seuil de classification, équilibrer les classes dans les données, ou appliquer une pénalisation asymétrique pour réduire les faux positifs.


## Conclusion

Ce projet avait pour objectif de prédire la potabilité de l'eau à partir de ses caractéristiques physico-chimiques à l’aide de techniques de machine learning supervisé. Après une phase de prétraitement rigoureux (imputation des valeurs manquantes, normalisation, séparation des jeux de données), un modèle Random Forest a été entraîné pour effectuer une classification binaire (potable vs non potable).
Le modèle a montré des performances satisfaisantes avec :

-une accuracy correcte,

-un bon score AUC, indiquant sa capacité à distinguer les deux classes,

-une matrice de confusion équilibrée,

-une interprétation des variables via l’importance des features.


L'intégration de MLflow a permis de tracer automatiquement les expériences, paramètres, modèles et métriques, ce qui constitue une bonne pratique en MLOps pour la reproductibilité et la comparaison des résultats.
Enfin, des visualisations clés (courbe ROC, importance des variables, matrice de confusion) ont renforcé l’analyse des performances et facilité l'interprétation des résultats.

