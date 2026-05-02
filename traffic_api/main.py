from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import os

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../pollution/traffic/traffic_dataset.csv')

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data")
def get_data():
    df = load_data()
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/api/summary")
def get_summary():
    df = load_data()
    summary = {
        "columns": list(df.columns),
        "count": len(df),
        "describe": df.describe(include='all').to_dict()
    }
    return summary

@app.post("/api/upload")
def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv(DATA_PATH, index=False)
        return {"status": "success", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
