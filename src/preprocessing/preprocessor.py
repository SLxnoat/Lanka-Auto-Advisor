import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

class VehiclePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'body_type', 'condition', 'registered', 'auction_grade', 'reg_series']
        
    def process(self,df):
        
        processed_df = df.copy()
        print("Starting preprocessing...")
        
        processed_df['listing_date'] =pd.to_datetime(processed_df['listing_date'], errors='coerce')
        processed_df['listing_month'] = processed_df['listing_date'].dt.month
        processed_df.drop('listing_date', axis=1, inplace=True)
        
        for col in self.categorical_cols:
            if col in processed_df.columns:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col} with {len(le.classes_)} unique values.")
                print("Preprocessing completed.")
        return processed_df
    
    def save_label_encoders(self, path='src/utils/encoders.pkl'):
        joblib.dump(self.label_encoders, path)
        print(f"Label encoders saved to {path}.")