from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    MAX_UPLOAD_MB: int = 10
    TEMP_DIR: str = "./tmp"
    GEMINI_API_KEY: str = "XXXXXXX"
    class Config:
        env_file = ".env"

settings = Settings()
Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)
