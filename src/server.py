from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import api

DIST = Path("frontend/dist")


@asynccontextmanager
async def lifespan(app):
    if not api.state:
        api.load_state()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/api", api.app)
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
