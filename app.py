from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/version")
def version_check():
    return {"version": "test-1.0.0"}
