from fastapi import Header, HTTPException 
from src.config.settings import settings

def validate_api_key(api_key: str=Header(...,alias="API-KEY")):
    if api_key!= settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
def validate_password(password: str=Header(...,alias="PASSWORD")):
    if password!= "igem.iit.edu/2025":
        raise HTTPException(status_code=401, detail="Invalid Password")