from fastapi import Header, HTTPException 
from src.config.settings import settings

def validate_api_key(api_key: str=Header(...,alias="API-KEY")):
    if api_key!= settings.API_KEY:
        raise HTTPException(status=401, detail="Invalid API Key")