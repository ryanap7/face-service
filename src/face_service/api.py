"""
HTTP API, reachable only from the backend. Images come in as the raw
request body (JPEG, PNG or WebP); nothing is stored.
"""
import hmac
from contextlib import asynccontextmanager
from queue import Empty

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from face_service import config
from face_service.engine import EnginePool, decode


def create_app(pool_factory=lambda: EnginePool(config.POOL_SIZE)) -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state['pool'] = pool_factory()
        yield

    app = FastAPI(title='SIMTELPAS Face Service', lifespan=lifespan, docs_url=None, redoc_url=None)

    def authorize(request: Request):
        if config.API_KEY and not hmac.compare_digest(
            request.headers.get('x-api-key', ''), config.API_KEY
        ):
            raise HTTPException(401, 'Invalid API key')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': config.MODEL_VERSION, 'pool': config.POOL_SIZE}

    @app.post('/v1/analyze')
    async def analyze(request: Request, liveness: bool = True):
        """
        The largest face in the image: where it is, whether it is usable
        (quality.issues), whether it is a live person (liveness) and, when
        usable, its embedding for matching.
        """
        authorize(request)
        body = await request.body()
        if not body:
            raise HTTPException(400, 'Empty image')
        if len(body) > config.MAX_IMAGE_BYTES:
            raise HTTPException(413, 'Image too large')
        image = decode(body)
        if image is None:
            raise HTTPException(400, 'Not a readable image')
        try:
            return await run_in_threadpool(
                state['pool'].run, lambda engine: engine.analyze(image, liveness)
            )
        except Empty:
            raise HTTPException(503, 'Busy, try again')

    return app


app = create_app()
