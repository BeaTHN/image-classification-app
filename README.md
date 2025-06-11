# image-classification-app
Projet de mémoire de master : classification d’images avec CNN.

# Prédiction de Races de Chiens avec Deep Learning
Bienvenue dans cette application Streamlit permettant de classer des races de chiens à partir d'images, à l’aide d’un modèle MobileNetV2 pré-entraîné et affiné pour 120 races canines.

---

## Objectifs

- Classifier des images de chiens en identifiant leur race parmi 120 catégories.
- Proposer une interface simple, intuitive et accessible via le web.
- Enregistrer les prédictions pour un historique consultable.
- Préparer l’application à des extensions futures (diagnostic santé, etc.).

---

## Démo en ligne

[Cliquez ici pour accéder à l'application Streamlit](https://) 

---

## Technologies utilisées

- **Python 3.10+**
- **Streamlit** (interface web)
- **TensorFlow / Keras** (modèle CNN)
- **MobileNetV2** (architecture de base, fine-tunée)
- **Pillow**, **OpenCV** (traitement d’images)
- **SQLite** (base de données locale)
- **Matplotlib** (graphiques)

---

## Arborescence du projet
├── app.py # Script principal Streamlit
├── model.keras # Modèle de classification sauvegardé
├── breed_list.json # Liste des races correspondantes
├── requirements.txt # Dépendances à installer
├── predictions.db # Base de données locale (SQLite)
└── README.md # Documentation du projet

---

---

## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/BeaTHN/image-classification-app.git
cd image-classification-app

### 2. Installer les dépendances
pip install -r requirements.txt

### 3. Lancer l'application
streamlit run app.py

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse http://localhost:8501

---

## Fonctionnalités principales
- Chargement d’image via interface intuitive
- Classification en temps réel via MobileNetV2
- Affichage des résultats : race prédite, taux de confiance, image de référence
- Feedback utilisateur sur la précision du modèle
- Historique des prédictions
- Gestion des utilisateurs (accès admin sécurisé)
- Section "Santé animale" (à venir)

---

## Données d'entraînement
Le modèle est entraîné sur une version nettoyée du Stanford Dogs Dataset, contenant plus de 20 000 images de chiens réparties sur 120 races.

---

## Déploiement (Streamlit Community Cloud)
Créez un dépôt GitHub avec ce projet.

Ajoutez tous les fichiers requis (app.py, model.keras, etc.).

Connectez-vous à https://streamlit.io/cloud.

Cliquez sur "New app", sélectionnez votre dépôt GitHub et lancez le déploiement.

---

## Licence
Ce projet est publié sous licence none. Voir le fichier LICENSE pour plus d’informations.

---

## Contributions
Les contributions sont les bienvenues ! Vous pouvez ouvrir une issue ou soumettre une pull request.

---

## Contact
Pour toute question ou suggestion :
Email : [beateicethione@gmail.com]

---



