from fastapi import FastAPI

app = FastAPI(title="SiteScore AI")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/score")
def score(lat: float, lon: float):
    # scor V1 (simplu, doar de test): mai mare dacă e aproape de Brașov centru
    # Piata Sfatului approx: 45.6427, 25.5887
    d = ((lat-45.6427)**2 + (lon-25.5887)**2) ** 0.5
    s = max(0, 100 - d*1000)  # scade cu distanța
    return {"score": round(s, 1)}
