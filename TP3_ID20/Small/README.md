# Name That LLM — Analyse et adaptation d'un notebook Kaggle

Projet réalisé dans le cadre d'un travail de Machine Learning. L'objectif est d'analyser le notebook Kaggle [*0.681 | Qwen3-Embedding LoRA + Qwen3.8-27B* de lightningv08](https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b/notebook), de repérer les étapes de la chaîne de traitement déjà présentes, puis d'en proposer une adaptation exécutable dans Google Colab.

La chaîne étudiée est : **Data → EDA → Prétraitement → Transformations → Machine Learning → Évaluation → Déploiement**.

L'adaptation reprend la logique de la branche **SVC4** du notebook d'origine : quatre représentations TF-IDF du texte (`char`, `word`, `shape` et `function`) combinées à un classificateur `LinearSVC`. Elle ne réentraîne pas les grands modèles Qwen3 et leurs adaptateurs LoRA.

## Notebook

Le fichier principal est :

`Name_That_LLM_Colab_Guide_Analyse_Amelioration_FR(1).ipynb`

Il contient l'analyse du notebook de référence, le chargement des données, le pipeline SVC4 adapté, les mesures de performance et une démonstration d'inférence avec Gradio.

## Exécution dans Google Colab

Aucune installation locale de Python ou de Jupyter n'est nécessaire. Un navigateur et un compte Google suffisent.

1. Ouvrir [Google Colab](https://colab.research.google.com/).
2. Dans **Fichier → Importer le notebook**, sélectionner le fichier `.ipynb` du dépôt (après l'avoir téléchargé). Il est également possible d'utiliser **Fichier → Ouvrir le notebook → GitHub** si le dépôt est public.
3. Cliquer sur **Se connecter**, en haut à droite, pour démarrer un environnement d'exécution Python.
4. Laisser le type d'exécution standard (**CPU**) : la branche SVC4 adaptée ne nécessite pas de GPU. Le notebook d'origine avec Qwen3/LoRA est un autre cas, beaucoup plus coûteux.
5. Exécuter les cellules dans l'ordre, du début à la fin, avec les boutons ▶ à gauche des cellules. Pour cette version du notebook, le menu **Exécution → Tout exécuter** est aussi utilisable une fois les paramètres vérifiés.

### Paramètres

Au début du notebook, conserver les valeurs suivantes pour reproduire l'expérience réalisée :

```python
TELECHARGER_COMPETITION = False
CHEMIN_CSV_MANUEL = ""
MAX_EXEMPLES_PAR_CLASSE = 250
SEED = 42
```

- `TELECHARGER_COMPETITION = False` : le pipeline adapté utilise le dataset public étendu, sans avoir besoin du fichier de test de la compétition.
- `CHEMIN_CSV_MANUEL = ""` : le téléchargement est automatique. Ce champ sert seulement si les données ont déjà été placées à un emplacement spécifique dans Colab ou Drive.
- `MAX_EXEMPLES_PAR_CLASSE = 250` : limite le nombre de textes utilisés par modèle pour maîtriser la consommation de ressources. Mettre `None` pour conserver tous les exemples disponibles ; les résultats et temps de calcul pourront changer.
- `SEED = 42` : fixe la graine aléatoire utilisée pour l'échantillonnage.

La première cellule installe les dépendances nécessaires : `kagglehub`, `scikit-learn`, `joblib`, `matplotlib` et `gradio`. Les bibliothèques telles que `pandas`, `numpy` et `scipy` sont également utilisées.

### Données

Le notebook télécharge automatiquement le dataset [*prvsiyan's Extended Name That LLM Dataset*](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) avec `kagglehub` et sélectionne le fichier `passages_clean.csv`.

Les colonnes principales sont :

| Colonne | Utilisation |
| --- | --- |
| `text` | Texte à classifier |
| `model` | Modèle de langage à identifier (variable cible) |
| `prompt_id` | Identifiant du prompt, utilisé pour séparer les ensembles sans partager les mêmes prompts |

Les autres métadonnées du dataset ne sont pas injectées directement dans le classificateur. En particulier, des champs comme `requested_model_id` ou `served_model_id` pourraient révéler l'étiquette recherchée.

Le téléchargement du dataset public a fonctionné sans authentification supplémentaire lors de l'exécution du projet. Si Kaggle demande une authentification, un jeton peut être créé dans [les paramètres du compte Kaggle](https://www.kaggle.com/settings/api) puis enregistré dans les **Secrets** de Colab sous `KAGGLE_API_TOKEN`. Il ne faut pas inclure ce jeton dans le notebook ni dans GitHub.

## Déroulement du pipeline

1. **Data** : chargement du CSV et vérification de ses colonnes.
2. **EDA** : effectifs par modèle, longueurs des textes, valeurs manquantes et doublons.
3. **Prétraitement** : retrait des textes vides et des doublons, puis découpage entraînement/validation/test en séparant les `prompt_id`.
4. **Transformations** : extraction des quatre vues TF-IDF, ajustées uniquement sur l'entraînement.
5. **Machine Learning** : apprentissage d'un `LinearSVC` et comparaison avec une baseline qui prédit toujours la classe majoritaire.
6. **Évaluation** : accuracy, balanced accuracy, F1-macro, rapport de classification et matrice de confusion.
7. **Déploiement** : sauvegarde du modèle et des vectoriseurs, rechargement et interface de prédiction Gradio.

## Résultats de l'exécution réalisée

Avec `MAX_EXEMPLES_PAR_CLASSE = 250`, le fichier `passages_clean.csv` contient initialement **5 070 textes** correspondant à **18 modèles**. Après échantillonnage, **4 500 textes** ont été utilisés : **3 192** pour l'entraînement, **678** pour la validation et **630** pour le test.

| Modèle et jeu | Accuracy | Balanced accuracy | F1-macro |
| --- | ---: | ---: | ---: |
| Baseline — validation | 0,0546 | 0,0556 | 0,0057 |
| SVC4 adapté — validation | 0,3289 | 0,3297 | 0,3116 |
| SVC4 adapté — test | 0,3222 | 0,3222 | 0,3026 |

Ces valeurs proviennent de l'exécution enregistrée dans le notebook joint. Elles peuvent évoluer si le dataset, les paramètres ou les versions des bibliothèques changent. Les scores ne sont **pas directement comparables au 0,681** du notebook Kaggle original, dont l'architecture et le protocole diffèrent.

## Tester l'interface de prédiction

Dans la dernière cellule Python, la variable suivante contrôle l'affichage de l'application :

```python
LANCER_INTERFACE = True
```

Après avoir entraîné et sauvegardé le modèle :

1. Exécuter la cellule contenant `gr.Interface(...)`.
2. Attendre l'affichage de **Name That LLM — SVC4 adapté** dans Colab.
3. Saisir un paragraphe en anglais dans le champ **Texte à analyser**.
4. Cliquer sur **Submit**. Le nom du LLM prédit apparaît dans **Modèle prédit**.

Il s'agit d'une **attribution statistique**, pas d'une preuve certaine de l'origine du texte. L'interface s'exécute tant que la session Colab reste active ; elle n'est pas hébergée en permanence.

## Sauvegarde des fichiers produits

Le modèle et ses vectoriseurs sont sauvegardés ensemble dans :

```text
/content/name_that_llm/sorties/svc4_adapte.joblib
```

Pour récupérer ce fichier sur son ordinateur, on peut ouvrir le panneau **Fichiers** de Colab, suivre le chemin ci-dessus, puis utiliser le menu du fichier pour le télécharger. Autre possibilité, exécuter cette cellule après l'entraînement :

```python
from google.colab import files
files.download(str(modele_path))
```

Les fichiers présents sous `/content` ne sont pas conservés automatiquement après la fin de la session Colab. Les données sources et les gros artefacts du modèle n'ont donc pas besoin d'être ajoutés au dépôt GitHub pour exécuter le notebook.

## Limites et provenance

La tentative de reproduction du notebook original est limitée par ses dépendances à la compétition (notamment `test.csv`), aux checkpoints externes et à des ressources GPU importantes. Dans l'exécution documentée, le test officiel était absent : **l'inférence complète Qwen3/LoRA n'a pas été reproduite**.

Le travail présenté dans ce dépôt consiste en un audit de ce notebook et en une adaptation reproductible de sa branche SVC4. Le code de l'ensemble original reste consultable depuis le lien Kaggle ci-dessous. Le fichier `.ipynb` fourni dans ce dépôt ne contient pas l'intégralité des 50 cellules originales en annexe.

### Sources

- [Notebook de référence — lightningv08](https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b/notebook)
- [Dataset étendu — prvsiyan](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset)
- [Documentation Google Colab](https://research.google.com/colaboratory/intl/fr/faq.html)
- [KaggleHub — bibliothèque Python](https://github.com/Kaggle/kagglehub)
