from pydantic_settings import BaseSettings, SettingsConfigDict  

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    GROQ_API_KEY: str
    API_KEY: str = "iGEM-IIT"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "Gemma2-9b-It"
    VECTOR_DB_DIR: str = "./vectorstores"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(extra="allow")  

settings = Settings()