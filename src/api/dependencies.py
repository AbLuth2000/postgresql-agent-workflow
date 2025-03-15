from fastapi import Depends, HTTPException, Header

API_KEY = "your-secret-api-key"

def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
