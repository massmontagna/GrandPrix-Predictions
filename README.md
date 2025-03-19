# Formula 1 Race Predictor

This project uses machine learning to predict Formula 1 race results based on qualifying session data. It implements two models: a Random Forest model and a Deep Learning model, which are combined to make ensemble predictions.

## Features

- Data gathering from FastF1 API
- Historical data preprocessing
- Two different prediction models:
  - Random Forest (simple ML)
  - Deep Learning (neural network)
- Ensemble predictions combining both models
- Support for different circuits and years
- Weather data integration
- Driver and team encoding

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, gather historical data:
```bash
python data_gathering.py
```

2. Train the models:
```bash
python model_training.py
```

3. Make predictions:
```bash
python predict.py
```

When running the prediction script, you'll be prompted to enter:
- The year of the race
- The race name (e.g., "Monaco Grand Prix")

The script will then fetch the qualifying data for that race and provide predictions for the race results.

## Project Structure

- `data_gathering.py`: Fetches historical F1 data using FastF1
- `data_preprocessing.py`: Preprocesses the data for model training
- `model_training.py`: Implements and trains both ML models
- `predict.py`: Makes predictions using the trained models
- `requirements.txt`: Lists all required Python packages
- `models/`: Directory where trained models and preprocessors are saved
- `cache/`: Directory for FastF1 cache

## Notes

- The models are trained on historical data from 2018-2023
- Weather data is included in the predictions
- The system handles different drivers and teams across seasons
- Models are saved after training and can be reused
