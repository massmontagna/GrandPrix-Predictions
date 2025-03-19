import pandas as pd
import numpy as np
from data_gathering import F1DataGatherer
from data_preprocessing import F1DataPreprocessor
from model_training import F1ModelTrainer
import joblib
import tensorflow as tf
from tensorflow.keras import models

class F1Predictor:
    def __init__(self):
        self.gatherer = F1DataGatherer()
        self.preprocessor = F1DataPreprocessor()
        self.trainer = F1ModelTrainer()
        
        # Load preprocessors and models
        self.preprocessor.load_preprocessors()
        self.trainer.load_models()
    
    def predict_race(self, year, race_name):
        """Predict race results using both models."""
        # Get qualifying data
        quali_data = self.gatherer.get_current_quali_data(year, race_name)
        if quali_data is None:
            print(f"Could not fetch qualifying data for {year} {race_name}")
            return None
        
        # Print initial data info
        print("\nInitial qualifying data:")
        print(f"Number of drivers: {len(quali_data)}")
        print(quali_data[['Driver', 'Team', 'QualiPosition']].head())
        
        # Preprocess data
        X_scaled, quali_data = self.preprocessor.preprocess_prediction_data(quali_data)
        
        # Print preprocessed data info
        print("\nPreprocessed data:")
        print(f"X_scaled shape: {X_scaled.shape}")
        print(f"Number of drivers in quali_data: {len(quali_data)}")
        
        # Make predictions with both models
        rf_predictions = self.trainer.rf_model.predict(X_scaled)
        dl_predictions = self.trainer.dl_model.predict(X_scaled)
        
        # Print prediction shapes
        print("\nPrediction shapes:")
        print(f"RF predictions shape: {rf_predictions.shape}")
        print(f"DL predictions shape: {dl_predictions.shape}")
        
        # Ensure both predictions are 1D arrays
        rf_predictions = rf_predictions.ravel()
        dl_predictions = dl_predictions.ravel()
        
        # Combine predictions (ensemble)
        ensemble_predictions = (rf_predictions + dl_predictions) / 2
        
        # Ensure all arrays have the same length
        driver_names = quali_data['Driver'].values
        team_names = quali_data['Team'].values
        quali_positions = quali_data['QualiPosition'].values
        
        print("\nArray lengths:")
        print(f"Driver names length: {len(driver_names)}")
        print(f"Team names length: {len(team_names)}")
        print(f"Quali positions length: {len(quali_positions)}")
        print(f"Ensemble predictions length: {len(ensemble_predictions)}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Driver': driver_names,
            'Team': team_names,
            'QualiPosition': quali_positions,
            'PredictedPosition': ensemble_predictions
        })
        
        # Sort by predicted position
        results = results.sort_values('PredictedPosition')
        
        # Add rank
        results['Rank'] = range(1, len(results) + 1)
        
        return results
    
    def print_predictions(self, predictions):
        """Print predictions in a formatted way."""
        if predictions is None:
            return
        
        print("\nPredicted Race Results:")
        print("=" * 80)
        print(f"{'Rank':<5} {'Driver':<25} {'Team':<25} {'Quali':<8} {'Predicted':<10}")
        print("-" * 80)
        
        for _, row in predictions.iterrows():
            print(f"{row['Rank']:<5} {row['Driver']:<25} {row['Team']:<25} "
                  f"{row['QualiPosition']:<8} {row['PredictedPosition']:.1f}")

def main():
    predictor = F1Predictor()
    
    # Get user input
    year = int(input("Enter the year: "))
    race_name = input("Enter the race name (e.g., 'Monaco Grand Prix'): ")
    
    # Make predictions
    predictions = predictor.predict_race(year, race_name)
    
    # Print results
    predictor.print_predictions(predictions)

if __name__ == "__main__":
    main() 