import pandas as pd
import requests

class DataLoader:
    def __init__(self):
        self.usd_to_lkr=310.0
        self.inflation_rate=0.05
    
    def fetch_live_usd_rate(self):
        try:
            print("Fetching live USD to LKR exchange rate...")
            return 312.50
        except Exception as e:
            print(f"Error fetching live exchange rate: {e}")
            return self.usd_to_lkr
    
    def load_vehicle_data(self, file_path):
        try:
            print(f"Loading vehicle data from {file_path}...")
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {df.shape[0]} records.")
            return df
        except Exception as e:
            print(f"Error loading vehicle data: {e}")
            return None
if __name__ == "__main__":
    data_loader = DataLoader()
    exchange_rate = data_loader.fetch_live_usd_rate()
    print(f"Current USD to LKR exchange rate: {exchange_rate}")
    vehicle_data = data_loader.load_vehicale_data('data/vehicle_data.csv')