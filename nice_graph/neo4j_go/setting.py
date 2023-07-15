import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path('.') / '.env'
env_path = env_path.resolve()
print(f"Your path with cre888ntials: {env_path}")
load_dotenv(dotenv_path=env_path)
NEO4J_USER = os.environ.get('NEO4J_USER')
NEO4J_PASSWORD = os.environ.get('PASSWORD')
print(f"NEO4J user:{NEO4J_USER}. password: {NEO4J_PASSWORD}")
