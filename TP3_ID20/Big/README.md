# Détection d'intrusions réseau avec Apache Spark — UNSW-NB15

## Présentation du projet

Dans ce projet, j'ai étudié un notebook Kaggle consacré à la détection de menaces réseau sur le dataset **UNSW-NB15**, puis j'ai développé une adaptation avec **PySpark et Spark MLlib**.

L'objectif de mon travail était de vérifier si le notebook d'origine respectait la chaîne **DATA → EDA → Prétraitement → Transformations → ML → EVAL → Deploy**, d'identifier ses points faibles et de proposer des améliorations en expliquant les choix effectués.

J'ai conservé le problème de cybersécurité, mais j'ai changé l'approche : le notebook Kaggle utilise un **réseau de neurones (ANN) pour une classification en 11 catégories**, tandis que mon adaptation utilise une **régression logistique Spark MLlib pour distinguer le trafic normal du trafic malveillant**. Les scores des deux modèles ne sont donc pas directement comparables.

## Sources

- **Notebook Kaggle analysé :** [UNSW_NB15 | Cybersecurity Threat Detection | ANN — Anmol Gupta](https://www.kaggle.com/code/getanmolgupta01/unsw-nb15-cybersecurity-threat-detection-ann)
- **Dataset :** [UNSW-NB15 sur Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Notebook de mon adaptation :** [`UNSW_NB15_TP3.ipynb`](UNSW_NB15_TP3.ipynb)

## 1. Analyse du notebook d'origine

J'ai examiné les **126 cellules** du notebook Kaggle fourni ainsi que leurs résultats enregistrés, afin d'identifier les étapes déjà réalisées et celles à compléter.

| Étape | État dans le notebook Kaggle | Constats |
|---|---|---|
| **DATA** | Présente, avec une limite | Chargement et concaténation des CSV avec Pandas ; les fichiers bruts sont lus sans `header=None`. |
| **EDA** | Présente | Statistiques descriptives, valeurs manquantes, doublons, visualisations et corrélations. |
| **Prétraitement** | Présent, mais perfectible | Conversions et nettoyage déjà réalisés ; certaines valeurs absentes sont remplacées par zéro sans indicateur d'absence. |
| **Transformations** | Présentes, mais protocole à corriger | Encodage, SMOTE, sélection de caractéristiques, standardisation et PCA ; certaines opérations sont appliquées avant la séparation entraînement/test. |
| **ML** | Présente | Réseau de neurones TensorFlow/Keras et recherche d'hyperparamètres. |
| **EVAL** | Partielle | Accuracy, loss et courbes ; le même ensemble est réutilisé pour la validation, le tuning et l'évaluation. |
| **Deploy** | Manquante | Pas de sauvegarde/rechargement du modèle ni de démonstration d'inférence sans étiquette. |

Le notebook original contient déjà une chaîne de traitement assez complète : mon travail ne consiste donc pas à prétendre que toutes les phases étaient absentes. J'ai surtout cherché à **corriger certaines limites méthodologiques, utiliser Spark pour le traitement du volume de données et ajouter une phase de réutilisation du modèle**.

L'ANN de l'auteur affiche une **accuracy de 70,26 %** dans ses sorties enregistrées, pour une tâche de classification à **11 catégories**. Je n'ai pas réexécuté cet ANN dans mon environnement Colab ; ce chiffre provient du notebook original.

## 2. Adaptation et améliorations

### DATA — Chargement avec PySpark

J'ai utilisé les **quatre fichiers CSV bruts** du dataset UNSW-NB15 plutôt que de me limiter aux fichiers d'entraînement/test préconstitués. Le chargement avec Spark, sans en-tête, donne **2 540 047 lignes et 49 colonnes**.

J'ai ensuite récupéré les noms des variables depuis le fichier de description des caractéristiques. L'exécution Kaggle d'origine affichait 2 540 043 lignes ; cette différence de quatre lignes est cohérente avec sa lecture des quatre fichiers sans préciser `header=None`. Le nombre souvent publié de 2 540 044 observations reste une divergence documentaire que je signale sans modifier arbitrairement les données.

### EDA — Comprendre le trafic réseau

J'ai étudié les protocoles les plus fréquents, les catégories d'attaques et la distribution des classes avec des agrégations Spark. Le dataset présente un **fort déséquilibre** :

- **2 218 764 connexions normales** (environ 87,35 %) ;
- **321 283 connexions malveillantes** (environ 12,65 %).

Ce déséquilibre explique pourquoi une accuracy élevée ne suffit pas pour juger un système de détection : il faut également regarder le **rappel des attaques** et les **faux négatifs**.

### Prétraitement — Nettoyage et valeurs manquantes

J'ai normalisé les catégories, corrigé un nom de colonne contenant un espace et converti les variables numériques avec `try_cast` pour pouvoir identifier les valeurs non convertibles.

Trois caractéristiques présentaient beaucoup de valeurs absentes :

| Caractéristique | Valeurs absentes |
|---|---:|
| `is_ftp_login` | 1 429 879 |
| `ct_ftp_cmd` | 1 429 879 |
| `ct_flw_http_mthd` | 1 348 145 |

Après avoir étudié leur répartition par service réseau, j'ai conservé ces variables et ajouté **un indicateur d'absence** pour chacune d'elles. J'ai ensuite remplacé les valeurs invalides par zéro, sans perdre l'information indiquant qu'une valeur était initialement manquante.

### Transformations — Parquet et pipeline MLlib

J'ai enregistré les données nettoyées en **Parquet avec compression Snappy**, puis construit un pipeline contenant :

1. `StringIndexer` pour les variables `proto`, `state` et `service` ;
2. `OneHotEncoder` pour les catégories ;
3. `VectorAssembler` pour réunir les caractéristiques ;
4. `StandardScaler` pour la mise à l'échelle ;
5. `LogisticRegression` pour la classification binaire.

Les étapes du pipeline sont ajustées sur les données d'entraînement, puis appliquées à la validation et au test. J'ai exclu `attack_cat` des variables explicatives, car cette colonne dévoile la nature normale ou malveillante d'une connexion.

### ML — Classification binaire

J'ai entraîné une régression logistique avec Spark MLlib, après une séparation en ensembles d'entraînement, de validation et de test. L'entraînement enregistré a duré **177,32 secondes** dans ma session Google Colab.

### EVAL — Mesurer la détection des attaques

L'évaluation finale sur **507 459 connexions de test** donne les résultats suivants :

| Métrique | Résultat |
|---|---:|
| Accuracy | **97,86 %** |
| Précision des attaques | **90,22 %** |
| Rappel des attaques | **93,21 %** |
| F1-score des attaques | **91,69 %** |
| PR-AUC | **98,64 %** |

**Matrice de confusion :**

| Classe réelle / prédite | Normal | Attaque |
|---|---:|---:|
| **Normal** | 436 552 | 6 508 |
| **Attaque** | 4 374 | 60 025 |

Le modèle identifie **60 025 des 64 399 attaques** présentes dans le test, mais en manque encore **4 374**. Dans un contexte de cybersécurité, ces faux négatifs restent un point important à améliorer.

### Deploy — Sauvegarde et inférence

J'ai sauvegardé le **pipeline complet**, puis vérifié son rechargement avec `PipelineModel.load()`. J'ai également simulé une prédiction sur des connexions dont la colonne `label` avait été retirée.

Sur une démonstration de dix exemples sélectionnés (cinq normaux et cinq malveillants), le modèle a correctement classé les cinq connexions normales et détecté quatre attaques sur cinq. Cet échantillon illustre l'inférence ; **il ne constitue pas une nouvelle mesure des performances**.

Il s'agit d'un **déploiement expérimental** dans Spark, et non d'une solution de surveillance réseau en production.

## 3. Technologies utilisées

- **Python** et **Google Colab** ;
- **Apache Spark / PySpark**, **Spark MLlib** ;
- **Pandas** (pour le dictionnaire des colonnes et les agrégats à visualiser) ;
- **Matplotlib** ;
- **KaggleHub** pour récupérer le dataset ;
- **Parquet / Snappy** pour le stockage intermédiaire.

Spark est utilisé dans Colab avec `local[2]` : les traitements reposent sur les API Spark, mais **je n'ai pas déployé de cluster Hadoop ou Spark multinœud**.

## 4. Exécuter le notebook

1. Ouvrir [`UNSW_NB15_TP3.ipynb`](UNSW_NB15_TP3.ipynb) dans **Google Colab**.
2. Ajouter le secret comme indiqué dans le ReadMe à la racine du dossier TP3.
3. Exécuter les cellules **dans l'ordre**, en commençant par l'installation de `kagglehub` et `pyspark` :

   ```python
   %pip install -q kagglehub pyspark
   ```

4. Laisser le notebook télécharger les données, démarrer Spark et exécuter l'EDA, le prétraitement et l'entraînement.
5. Examiner la validation, les résultats du test final et la démonstration d'inférence.

Le dataset est téléchargé automatiquement avec `kagglehub.dataset_download("mrwellsdavid/unsw-nb15")`. Une connexion Internet et des ressources mémoire/CPU suffisantes sont nécessaires. Les chemins `/content/unsw_nb15_parquet` et `/content/unsw_nb15_spark_model` pointent vers le **stockage temporaire de Colab** ; ils ne persistent pas après la fermeture ou la réinitialisation de la session, sauf export explicite.

Les dernières cellules du notebook contiennent un **contrôle complémentaire des doublons** et une **réévaluation facultative par groupes**. Ces cellules demandent un réentraînement et n'ont pas produit les métriques de référence ci-dessus.

## 5. Limites et pistes d'amélioration

Mon adaptation apporte une chaîne Spark reproductible dans l'environnement prévu, mais plusieurs limites demeurent :

- **Doublons et observations corrélées :** mon premier découpage est aléatoire sur les lignes brutes. Des caractéristiques identiques ou des flux proches peuvent être présents à la fois dans l'entraînement et le test. J'ai ajouté un contrôle des empreintes de caractéristiques et un protocole alternatif par groupes, mais **je n'ai pas encore de nouveaux résultats validés pour ce protocole**.
- **Comparaison des modèles :** l'ANN original prédit 11 catégories, tandis que mon modèle distingue deux classes. Les accuracies ne mesurent donc pas le même problème.
- **Détection imparfaite :** le modèle produit encore des fausses alertes et manque certaines attaques.
- **Généralisation :** une séparation temporelle, une évaluation sur de nouveaux environnements réseau et une validation après dédoublonnage seraient plus représentatives d'un usage réel.
- **Déploiement :** l'inférence fonctionne dans Spark, mais il manque une ingestion de trafic en temps réel, des mécanismes de supervision et un hébergement persistant.

## Conclusion

Ce travail m'a permis de **partir d'un notebook existant, d'en analyser les étapes et les limites, puis de proposer une adaptation orientée Big Data**. L'apport principal est l'utilisation de Spark sur les quatre fichiers bruts, la gestion explicite de certaines valeurs manquantes, l'intégration des transformations dans un pipeline MLlib et l'ajout d'une phase de sauvegarde et d'inférence.

Les résultats obtenus sont encourageants pour une première classification binaire, mais je les présente comme ceux de **mon protocole expérimental actuel**, avec les réserves liées aux doublons et à la généralisation.
