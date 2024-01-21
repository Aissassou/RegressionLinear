import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Création d'une base de données pour tester
players_data = [ 
    {'Nom': 'Joueur1', 'Age': 18, 'Matches_joues': 50, 'Degre_blessure': 0.2},
    {'Nom': 'Joueur2', 'Age': 25, 'Matches_joues': 45, 'Degre_blessure': 0.4},
    {'Nom': 'Joueur3', 'Age': 18, 'Matches_joues': 60, 'Degre_blessure': 0.1},
    {'Nom': 'Joueur4', 'Age': 28, 'Matches_joues': 55, 'Degre_blessure': 0.6},
    {'Nom': 'Joueur5', 'Age': 17, 'Matches_joues': 28, 'Degre_blessure': 0.0},
    {'Nom': 'Joueur6', 'Age': 32, 'Matches_joues': 10, 'Degre_blessure': 0.2},
    {'Nom': 'Joueur7', 'Age': 22, 'Matches_joues': 40, 'Degre_blessure': 0.3},
    {'Nom': 'Joueur8', 'Age': 29, 'Matches_joues': 58, 'Degre_blessure': 0.5},
    {'Nom': 'Joueur9', 'Age': 31, 'Matches_joues': 52, 'Degre_blessure': 0.8},
    {'Nom': 'Joueur10', 'Age': 27, 'Matches_joues': 47, 'Degre_blessure': 0.9},
]

class FootballInjuryModelTrainer:
    def __init__(self):
        np.random.seed(42)
        self.data = self.generate_data()
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def generate_data(self):
        data = []
        for i in range(100):
            age = np.random.randint(18, 40)
            matches_joues = np.random.randint(20, 100)
        
            # Introduire une relation positive entre l'âge et le degré de blessure
            degre_blessure = np.random.uniform(0.0, 1.0) + 0.1 * age/40 - 0.1 * matches_joues/100

            # Assurer que le degré de blessure reste dans la plage [0, 1]
            degre_blessure = max(0, min(1, degre_blessure))

            player = {
            'Nom': f'Joueur{i + 1}',
            'Age': age,
            'Matches_joues': matches_joues,
            'Degre_blessure': degre_blessure
            }
            data.append(player)
        return pd.DataFrame(data)
#définition des differentes données pour l'entrainement
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[['Age', 'Matches_joues']],
            self.data['Degre_blessure'],
            test_size=0.2,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self):
        # Créer un modèle de régression linéaire
        self.model = LinearRegression()

        # Entrainement du model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Faire des prédictions sur les tests
        predictions = self.model.predict(self.X_test)

        # Evaluation  des performance du modèle si demander
        mse = mean_squared_error(self.y_test, predictions)

        ##print(f"Erreur quadratique moyenne du modèle : {mse}")
    
    def classify_players(self):
        player_data = pd.DataFrame(players_data)
        predicted_degrees = self.model.predict(player_data[['Age', 'Matches_joues']])
        predictions_labels = np.where(predicted_degrees > 0.5, 'Possible blessure', 'Pas de blessure') ##si on veut changer sinon mettre la ligne en commentaire pour afficher le pourcentage

        # Ajouter la colonne de prédictions 
        player_data['Real_Degre_blessure'] = player_data['Degre_blessure']
        player_data['Predicted_Degre_blessure'] = predictions_labels  

        # Afficher les résultats avec les prédictions
        print("Joueurs classés avec les prédictions du modèle :\n", player_data[['Nom', 'Real_Degre_blessure', 'Predicted_Degre_blessure']])

# Utilisation de la classe pour créer et entraîner le modèle et l'appeler
trainer = FootballInjuryModelTrainer()
trainer.train_model()
trainer.evaluate_model()
trainer.classify_players()
