import xgboost as xgb
import numpy as np
import pickle

# Load scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load saved model
xgb_loaded = xgb.Booster()
xgb_loaded.load_model("model/xgboost_model.json")

def infer_churn(user_input):
    """
    Predict churn probability and class using XGBoost model.
    user_input: list of numeric feature values
    """
    # Ensure correct shape
    new_data = np.array(user_input).reshape(1, -1)
    
    # Scale before converting to DMatrix
    scaled_data = scaler.transform(new_data)
    
    # Create DMatrix
    dnew = xgb.DMatrix(scaled_data)
    
    # Predict probability
    prob = xgb_loaded.predict(dnew)[0]
    
    return {"probability": float(prob), "class": int(prob > 0.5)}

# Example usage:
# print(infer_churn([650, 35, 3, 15000, 2, 1, 1, 50000, 0, 1]))
