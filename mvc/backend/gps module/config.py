import os

from dotenv import load_dotenv
load_dotenv()

class Config:
    #MONGO_URI = os.getenv("MONGO_URI")
    MONGO_URI = "postgresql://postgres:1234@localhost:5432/postgres"
    print(MONGO_URI)
