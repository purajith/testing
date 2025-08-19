import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from metrics import evaluate_metrics
from data_preprocessing import data_processing, scaling
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# load the lebel encoder 
label_encoder = LabelEncoder()

# model training with multiple algorithms
def train_models(x_train, x_test, y_train, y_test):
    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(x_train, y_train)
    models['Random Forest'] = (rf, rf.predict(x_test), rf.predict_proba(x_test)[:, 1])
    # rs

    param_dist = {
        
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
    }
    rs = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, scoring='recall', cv=3)
    rs.fit(x_train, ytrain)
    models['Random search'] = (rs, rs.predict(x_test), rs.predict_proba(x_test)[:, 1])

    # XGBoost
    xgb_train = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    xgb_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    xgb_model = xgb.train(params={'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1},
                          dtrain=xgb_train, num_boost_round=50)
    xgb_preds = xgb_model.predict(xgb_test)
    models['XGBoost'] = (xgb_model, (xgb_preds > 0.5).astype(int), xgb_preds)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    lgb_model.fit(x_train, y_train)
    models['LightGBM'] = (lgb_model, lgb_model.predict(x_test), lgb_model.predict_proba(x_test)[:, 1])

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=16)
    lr = LogisticRegression(class_weight='balanced')

    lr.fit(x_train, y_train)
    models['Logistic Regression'] = (lr, lr.predict(x_test), lr.predict_proba(x_test)[:, 1])

    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    models['K-Nearest Neighbors'] = (knn, knn.predict(x_test), knn.predict_proba(x_test)[:, 1])

    # Save model to a file
    xgb_model.save_model("xgboost_model.json")  # Can also use .bin format
    print("Model saved as xgboost_model.json")
    return models


if __name__=="__main__":
    # load dataset
    file = pd.read_csv("data/Churn_Modelling.csv")
    file = data_processing(file)
    
    # scaling 
    scaled_data_x,y =  scaling(file)
    
    # train test split
    x_train, x_test, ytrain, ytest = train_test_split(scaled_data_x,y, test_size =0.2, stratify =y, random_state =42)
    
    # algorithm
    model = train_models(x_train, x_test, ytrain, ytest)
    # evaluation metrics 
    evaluate_metrics(model, ytest)

