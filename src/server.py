from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import api

DIST = Path("frontend/dist")

# A mounted sub-app does not run its own startup/shutdown, so the outer app
# reuses the API's lifespan (it opens and closes the database connection pool).
app = FastAPI(lifespan=api.lifespan)
app.mount("/api", api.app)
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
