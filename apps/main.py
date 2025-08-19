import logging
from fastapi import FastAPI
from pydantic import BaseModel
from apps.infer import infer_churn

# ----------------- LOGGING CONFIGURATION -----------------
logging.basicConfig(
    filename="logs/app.log",  # Log file path
    level=logging.INFO,       # Logging level (DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

app = FastAPI()

# ----------------- DATA MODELS -----------------
class login(BaseModel):
    userid: str
    password: str

class inputs(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: int
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    stimatedSalary: int
    Geography_encoded: int
    Gender_encoded: int

# ----------------- ROUTES -----------------
@app.get("/")
def home():
    logging.info("Home endpoint accessed")
    return "You are in bank customer churn predictions."

@app.post("/login")
def login(user_login: login):
    try:
        logging.info(f"Login attempt for UserID: {user_login.userid}")
        result = (
            "You logged in successfully"
            if user_login.userid == "1234" and user_login.password == "1234"
            else "Your credential is wrong"
        )
        logging.info(f"Login result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during login: {str(e)}")
        return {"error": "An unexpected error occurred during login."}

@app.post("/input")
def user_input(input: inputs):
    try:
        values = list(input.dict().values())
        logging.info(f"Received input values: {values}")

        result = infer_churn(values)
        logging.info(f"Prediction result: {result}")

        return {"result": result}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": "An unexpected error occurred during prediction."}
