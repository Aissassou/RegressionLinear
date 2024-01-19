import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Création d'une base de données factice pour tester
players_data = [ 
    {'Nom': 'Joueur1', 'Age': 25, 'Matches_joues': 50, 'Degre_blessure': 0.2},
    {'Nom': 'Joueur2', 'Age': 28, 'Matches_joues': 45, 'Degre_blessure': 0.4},
    {'Nom': 'Joueur3', 'Age': 23, 'Matches_joues': 60, 'Degre_blessure': 0.1},
    {'Nom': 'Joueur4', 'Age': 30, 'Matches_joues': 55, 'Degre_blessure': 0.6},
    {'Nom': 'Joueur5', 'Age': 17, 'Matches_joues': 28, 'Degre_blessure': 0.0},
    {'Nom': 'Joueur6', 'Age': 32, 'Matches_joues': 53, 'Degre_blessure': 0.0},
    {'Nom': 'Joueur7', 'Age': 22, 'Matches_joues': 40, 'Degre_blessure': 0.3},
    {'Nom': 'Joueur8', 'Age': 29, 'Matches_joues': 58, 'Degre_blessure': 0.5},
    {'Nom': 'Joueur9', 'Age': 31, 'Matches_joues': 52, 'Degre_blessure': 0.7},
    {'Nom': 'Joueur10', 'Age': 26, 'Matches_joues': 47, 'Degre_blessure': 0.9},
]

class FootballInjuryModelTrainer:
    def __init__(self):
        np.random.seed(42)
        self.data = self.generate_data()
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def generate_data(self):
        data = []
        for i in range(1000):
            player = {
                'Nom': f'Joueur{i + 1}',
                'Age': np.random.randint(18, 30),
                'Matches_joues': np.random.randint(20, 100),
                'Degre_blessure': np.random.uniform(0.0, 1.0)
            }
            data.append(player)
        return pd.DataFrame(data)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[['Age', 'Matches_joues', 'Degre_blessure']],
            (self.data['Degre_blessure'] > 0.5).astype(int),  # Utiliser 0.5 comme seuil pour la classification binaire
            test_size=0.2,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self):
        # Création du modèle de régression logistique
        self.model = LogisticRegression()

        # Entrainement du model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Prédictions sur les tests
        predictions = self.model.predict(self.X_test)

        # Evaluation des performance du modèle juste à titre indicatif
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        print(f"Précision du modèle : {accuracy}")
        print("Rapport de classification :\n", report)
    
    def classify_players(self):
        player_data = pd.DataFrame(players_data)
        class_predictions = self.model.predict(player_data[['Age', 'Matches_joues', 'Degre_blessure']])
        class_predictions_labels = np.where(class_predictions == 1, 'Possible blessure', 'Pas de blessure')
        # Ajouter la colonne de prédictions au DataFrame d'origine
        player_data['Real_Degre_blessure'] = player_data['Degre_blessure']
        player_data['Predicted_Injury'] = class_predictions_labels

        # Afficher les résultats avec les prédictions
        print("Joueurs classés avec les prédictions du modèle :\n", player_data[['Nom', 'Real_Degre_blessure', 'Predicted_Injury']])

# Utilisation de la classe pour créer et entraîner le modèle et l'appeler
trainer = FootballInjuryModelTrainer()
trainer.train_model()
trainer.evaluate_model()
trainer.classify_players()
