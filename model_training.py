import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os
from data_preprocessing import F1DataPreprocessor

class F1ModelTrainer:
    def __init__(self):
        self.preprocessor = F1DataPreprocessor()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.dl_model = None
        
    def build_dl_model(self, input_shape):
        """Build the deep learning model architecture."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def train_models(self):
        """Train both the Random Forest and Deep Learning models."""
        # Preprocess data
        X, y = self.preprocessor.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        print(f"Random Forest MSE: {rf_mse}")
        
        # Train Deep Learning
        print("Training Deep Learning model...")
        self.dl_model = self.build_dl_model((X_train.shape[1],))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.dl_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        dl_pred = self.dl_model.predict(X_test)
        dl_mse = mean_squared_error(y_test, dl_pred)
        print(f"Deep Learning MSE: {dl_mse}")
        
        # Save models
        self.save_models()
        
        return history
    
    def save_models(self):
        """Save both trained models."""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        joblib.dump(self.rf_model, 'models/rf_model.joblib')
        self.dl_model.save('models/dl_model.keras')
    
    def load_models(self):
        """Load both trained models."""
        self.rf_model = joblib.load('models/rf_model.joblib')
        self.dl_model = models.load_model('models/dl_model.keras')

if __name__ == "__main__":
    trainer = F1ModelTrainer()
    history = trainer.train_models()
    print("Model training completed!") 