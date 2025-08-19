import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from metrics import evaluate_metrics
import pickle
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Data preprocessing
def data_processing(file):
    
    # remove unwanted columns
    file = file.drop(["RowNumber","CustomerId","Surname"], axis =1, errors = "ignore")
    
    # drop fuplicates 
    file.dropna(inplace=True)

    # Drop duplicates
    file.drop_duplicates(inplace=True)
     # label encoding 

    # Fit and transform the Geography column
    file['Geography_encoded'] = label_encoder.fit_transform(file['Geography'])
    file['Gender_encoded'] = label_encoder.fit_transform(file['Gender'])

    # select major features
    file = file[['CreditScore',  'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Exited', 'Geography_encoded', 'Gender_encoded']]
    return file
    
# feature scaling 
def scaling(file):
    
    y = file["Exited"]
    X = file.drop(columns=["Exited"])
    
    # min max scalar 
    scaler = StandardScaler()
    model_ss = scaler.fit(X)
    scaled_data_x = model_ss.transform(X)
        # Save the fitted scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return scaled_data_x,y
