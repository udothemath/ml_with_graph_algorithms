import os
from pathlib import Path

from dotenv import load_dotenv

file_path = Path(__file__).resolve().parent
env_path = os.path.join(file_path, '.env')
print(f"Your path with credential: {env_path}")
load_dotenv(dotenv_path=env_path)
NEO4J_USER = os.environ.get('NEO4J_USER')
NEO4J_PASSWORD = os.environ.get('PASSWORD')
print(f"NEO4J user:{NEO4J_USER}. password: {NEO4J_PASSWORD}")
