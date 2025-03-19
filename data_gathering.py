import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

class F1DataGatherer:
    def __init__(self):
        self.cache_dir = 'cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        fastf1.Cache.enable_cache(self.cache_dir)
        
    def get_weather_data(self, session):
        """Extract weather data from a session, with fallback values if data is missing."""
        try:
            weather_data = session.weather_data
            if weather_data is not None and not weather_data.empty:
                # Try different possible column names
                temp_col = next((col for col in weather_data.columns if 'temperature' in col.lower()), None)
                humidity_col = next((col for col in weather_data.columns if 'humidity' in col.lower()), None)
                pressure_col = next((col for col in weather_data.columns if 'pressure' in col.lower()), None)
                
                # Print available columns for debugging
                print(f"Available weather columns: {weather_data.columns.tolist()}")
                
                # Use found columns or default values
                return {
                    'AvgTemperature': weather_data[temp_col].mean() if temp_col else 20.0,
                    'AvgHumidity': weather_data[humidity_col].mean() if humidity_col else 50.0,
                    'AvgPressure': weather_data[pressure_col].mean() if pressure_col else 1013.25
                }
        except Exception as e:
            print(f"Warning: Could not extract weather data: {str(e)}")
            if hasattr(session, 'weather_data'):
                print(f"Weather data type: {type(session.weather_data)}")
                if isinstance(session.weather_data, pd.DataFrame):
                    print(f"Weather data columns: {session.weather_data.columns.tolist()}")
        
        # Default values if weather data is not available
        return {
            'AvgTemperature': 20.0,  # Default temperature in Celsius
            'AvgHumidity': 50.0,     # Default humidity percentage
            'AvgPressure': 1013.25   # Default pressure in hPa
        }
        
    def get_race_data(self, year, race_name):
        """Fetch race and qualifying data for a specific race."""
        try:
            print(f"\nAttempting to fetch data for {year} {race_name}...")
            
            # First try to get the race session
            try:
                race = fastf1.get_session(year, race_name, 'R')
                race.load()
                print(f"Successfully loaded race session for {year} {race_name}")
            except Exception as e:
                print(f"Error loading race session: {str(e)}")
                return None
            
            # Then try to get the qualifying session
            try:
                quali = fastf1.get_session(year, race_name, 'Q')
                quali.load()
                print(f"Successfully loaded qualifying session for {year} {race_name}")
            except Exception as e:
                print(f"Error loading qualifying session: {str(e)}")
                return None
            
            # Get race results
            race_results = race.results[['DriverNumber', 'FullName', 'TeamName', 'Position']]
            race_results.columns = ['DriverNumber', 'Driver', 'Team', 'RacePosition']
            
            # Get qualifying results
            quali_results = quali.results[['DriverNumber', 'Position']]
            quali_results.columns = ['DriverNumber', 'QualiPosition']
            
            # Merge race and qualifying results
            results = pd.merge(race_results, quali_results, on='DriverNumber')
            
            # Add weather data
            weather_data = self.get_weather_data(race)
            for key, value in weather_data.items():
                results[key] = value
            
            print(f"Successfully processed data for {year} {race_name}")
            return results
            
        except Exception as e:
            print(f"Error processing data for {year} {race_name}: {str(e)}")
            return None
    
    def gather_historical_data(self, start_year=2022, end_year=2023):
        """Gather data for multiple races across years."""
        all_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"\nFetching schedule for year {year}...")
            try:
                schedule = fastf1.get_event_schedule(year)
                print(f"Found {len(schedule)} races for {year}")
                
                for _, race in schedule.iterrows():
                    race_name = race['EventName']
                    print(f"\nProcessing {year} {race_name}...")
                    
                    race_data = self.get_race_data(year, race_name)
                    if race_data is not None:
                        race_data['Year'] = year
                        race_data['RaceName'] = race_name
                        all_data.append(race_data)
                        print(f"Added data for {year} {race_name} to collection")
                    else:
                        print(f"Skipping {year} {race_name} due to missing data")
                        
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue
        
        if not all_data:
            print("\nNo data was successfully gathered!")
            return None
            
        print(f"\nSuccessfully gathered data for {len(all_data)} races")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Save the data
        combined_data.to_csv('historical_data.csv', index=False)
        print(f"\nSaved {len(combined_data)} total records to historical_data.csv")
        return combined_data
    
    def get_current_quali_data(self, year, race_name):
        """Fetch qualifying data for a specific race."""
        try:
            print(f"\nFetching qualifying data for {year} {race_name}...")
            quali = fastf1.get_session(year, race_name, 'Q')
            quali.load()
            print(f"Successfully loaded qualifying session")
            
            quali_results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Position']]
            quali_results.columns = ['DriverNumber', 'Driver', 'Team', 'QualiPosition']
            
            # Add weather data
            weather_data = self.get_weather_data(quali)
            for key, value in weather_data.items():
                quali_results[key] = value
            
            quali_results['Year'] = year
            quali_results['RaceName'] = race_name
            
            print(f"Successfully processed qualifying data for {year} {race_name}")
            return quali_results
            
        except Exception as e:
            print(f"Error fetching qualifying data for {year} {race_name}: {str(e)}")
            return None

if __name__ == "__main__":
    gatherer = F1DataGatherer()
    print("Starting data gathering process...")
    historical_data = gatherer.gather_historical_data()
    if historical_data is not None:
        print("\nData gathering completed successfully!")
    else:
        print("\nData gathering failed!") 