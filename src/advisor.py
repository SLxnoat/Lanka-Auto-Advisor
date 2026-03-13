import numpy as np

class VehicalAdvisor:
    def __init__(self, model_engine, preprocessor):
        self.model_engine = model_engine
        self.preprocessor = preprocessor
    
    def get_advice(self,user_input_df,listed_price):
        print("Processing user input for advice...")
        predicted_price = self.model_engine.predict(user_input_df)[0]
        
        lower_bound = predicted_price * 0.95
        upper_bound = predicted_price * 1.05
        
        if listed_price < lower_bound:
            status = "Good Deal"
            color = "green"
            negotiation = "Buy Now, it's a great deal!"
        
        elif listed_price<=listed_price<=upper_bound:
            status = "Fair Price"
            color = "orange"
            negotiation = "Consider negotiating for a better price."
        
        else:
            status = "Overpriced"
            color = "red"
            negotiation = "Price is too high. Negotiate hard or look for alternatives."
        
        return {
            "predicted_price": int(predicted_price),
            "status": status,
            "color": color,
            "advice": negotiation,
            "margin": int(listed_price - predicted_price)
        }