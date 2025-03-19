import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class F1DataPreprocessor:
    def __init__(self):
        self.driver_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data_path='historical_data.csv'):
        """Preprocess the historical data for model training."""
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        
        # Print initial data info
        print("\nInitial data info:")
        print(df.info())
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Remove rows with missing values in target variable
        df = df.dropna(subset=['RacePosition'])
        print(f"\nAfter removing rows with missing RacePosition: {len(df)} records")
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        df['DriverEncoded'] = self.driver_encoder.fit_transform(df['Driver'])
        df['TeamEncoded'] = self.team_encoder.fit_transform(df['Team'])
        df['CircuitEncoded'] = self.circuit_encoder.fit_transform(df['RaceName'])
        
        # Create feature matrix
        features = [
            'DriverEncoded', 'TeamEncoded', 'CircuitEncoded',
            'QualiPosition', 'AvgTemperature', 'AvgHumidity', 'AvgPressure'
        ]
        
        # Check for missing values in features
        print("\nMissing values in features:")
        print(df[features].isnull().sum())
        
        # Fill any remaining missing values in features with appropriate defaults
        df['QualiPosition'] = df['QualiPosition'].fillna(df['QualiPosition'].median())
        df['AvgTemperature'] = df['AvgTemperature'].fillna(20.0)
        df['AvgHumidity'] = df['AvgHumidity'].fillna(50.0)
        df['AvgPressure'] = df['AvgPressure'].fillna(1013.25)
        
        X = df[features]
        y = df['RacePosition']
        
        # Verify no NaN values remain
        if X.isnull().any().any() or y.isnull().any():
            print("\nWarning: NaN values still present in data!")
            print("X NaN values:", X.isnull().sum())
            print("y NaN values:", y.isnull().sum())
            raise ValueError("Data still contains NaN values after preprocessing")
        
        # Scale features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Save preprocessors
        self.save_preprocessors()
        
        print("\nPreprocessing completed successfully!")
        print(f"Final dataset shape: X: {X_scaled.shape}, y: {y.shape}")
        
        return X_scaled, y
    
    def preprocess_prediction_data(self, quali_data):
        """Preprocess qualifying data for prediction."""
        # Create a copy to avoid SettingWithCopyWarning
        quali_data = quali_data.copy()
        
        # Handle unseen drivers
        unseen_drivers = set(quali_data['Driver']) - set(self.driver_encoder.classes_)
        if unseen_drivers:
            print(f"\nFound new drivers: {unseen_drivers}")
            # Add new drivers to the encoder
            all_drivers = list(self.driver_encoder.classes_) + list(unseen_drivers)
            self.driver_encoder.fit(all_drivers)
        
        # Handle unseen teams
        unseen_teams = set(quali_data['Team']) - set(self.team_encoder.classes_)
        if unseen_teams:
            print(f"Found new teams: {unseen_teams}")
            # Add new teams to the encoder
            all_teams = list(self.team_encoder.classes_) + list(unseen_teams)
            self.team_encoder.fit(all_teams)
        
        # Handle unseen circuits
        unseen_circuits = set(quali_data['RaceName']) - set(self.circuit_encoder.classes_)
        if unseen_circuits:
            print(f"Found new circuits: {unseen_circuits}")
            # Add new circuits to the encoder
            all_circuits = list(self.circuit_encoder.classes_) + list(unseen_circuits)
            self.circuit_encoder.fit(all_circuits)
        
        # Encode categorical variables
        quali_data['DriverEncoded'] = self.driver_encoder.transform(quali_data['Driver'])
        quali_data['TeamEncoded'] = self.team_encoder.transform(quali_data['Team'])
        quali_data['CircuitEncoded'] = self.circuit_encoder.transform(quali_data['RaceName'])
        
        # Create feature matrix
        features = [
            'DriverEncoded', 'TeamEncoded', 'CircuitEncoded',
            'QualiPosition', 'AvgTemperature', 'AvgHumidity', 'AvgPressure'
        ]
        
        # Fill any missing values in features
        quali_data['QualiPosition'] = quali_data['QualiPosition'].fillna(quali_data['QualiPosition'].median())
        quali_data['AvgTemperature'] = quali_data['AvgTemperature'].fillna(20.0)
        quali_data['AvgHumidity'] = quali_data['AvgHumidity'].fillna(50.0)
        quali_data['AvgPressure'] = quali_data['AvgPressure'].fillna(1013.25)
        
        X = quali_data[features]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, quali_data
    
    def save_preprocessors(self):
        """Save the preprocessors for later use."""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        joblib.dump(self.driver_encoder, 'models/driver_encoder.joblib')
        joblib.dump(self.team_encoder, 'models/team_encoder.joblib')
        joblib.dump(self.circuit_encoder, 'models/circuit_encoder.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
    
    def load_preprocessors(self):
        """Load the preprocessors."""
        self.driver_encoder = joblib.load('models/driver_encoder.joblib')
        self.team_encoder = joblib.load('models/team_encoder.joblib')
        self.circuit_encoder = joblib.load('models/circuit_encoder.joblib')
        self.scaler = joblib.load('models/scaler.joblib')

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    preprocessor = F1DataPreprocessor()
    X, y = preprocessor.preprocess_data()
    print("Data preprocessing completed!") 