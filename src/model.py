import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import joblib
import os

class VehiclePriceModel:
    def __init__(self, model_path='src/models/vehicle_price_model.pkl'):
        self.model_path = model_path
        self.model = None
        
    def train(self, df):
        print("Starting model training...")
        
        X = df.drop('listed_price_lkr', axis=1)
        y = df['listed_price_lkr']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        
        self.model = xg.XGBRegressor(n_estimators=1000,learning_rate=0.05, max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
        self.model.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=False)
        print("Model training completed.")
        
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"📉 Model Evaluation - MAE: Rs. {mae:,.2f}")
        print(f"R-Squared Score: {r2:.4f}")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}.")
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}.")
            return True
        return False
    
    def predict(self, input_data):
        if self.model:
            return self.model.predict(input_data)
        return None