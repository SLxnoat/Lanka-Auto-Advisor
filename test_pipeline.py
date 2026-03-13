from src.data_loader import DataLoader
from src.preprocessing.preprocessor import VehiclePreprocessor
from src.model import VehiclePriceModel

loader = DataLoader()
df = loader.load_vehicle_data('data/vehicle_data.csv')

if df is not None:
    preprocessor = VehiclePreprocessor()
    clean_df = preprocessor.process(df)
    
    print("\n--- Cleaned Data Preview ---")
    print(clean_df.head())
    print(f"\nData shape after preprocessing: {clean_df.shape}")
    
    print("\n--- Starting Model Training ---")
    model_engine = VehiclePriceModel()
    model_engine.train(clean_df)