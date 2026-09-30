import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import Field

from adaptlm.config import Settings
from adaptlm.contracts.schema import SCHEMA_VERSION, StrictModel
from adaptlm.data.pipeline import read_rows
from adaptlm.inference.service import InferenceService

WEB = Path(__file__).parents[1] / "web"


class TriageRequest(StrictModel):
    message: str = Field(min_length=1, max_length=20000)
    mode: Literal["fixture", "zero_shot", "few_shot", "adapted"] = "fixture"


class CompareRequest(StrictModel):
    message: str = Field(min_length=1, max_length=20000)


def create_app(settings: Settings | None = None, engine=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        app.state.startup_error = None
        try:
            app.state.service = await asyncio.to_thread(InferenceService, settings, engine)
        except Exception as exc:
            app.state.startup_error = type(exc).__name__
        yield

    app = FastAPI(title="AdaptLM", version="1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    templates = Jinja2Templates(directory=WEB / "templates")

    @app.middleware("http")
    async def request_envelope(request, call_next):
        request.state.request_id = str(uuid4())
        # Starlette/Pydantic would otherwise read an arbitrarily large request before validation.
        if request.method in ("POST", "PUT", "PATCH"):
            size = 0
            chunks = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > 100000:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "invalid_input",
                                "detail": "request body exceeds limit",
                            },
                            "request_id": request.state.request_id,
                        },
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request.state.request_id,
                "error": {"code": "request_failed", "detail": str(exc.detail)},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo the user's text or credentials in validation errors.
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request.state.request_id,
                "error": {
                    "code": "invalid_input",
                    "detail": "request does not match the API contract",
                },
            },
        )

    def admin(request: Request):
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else ""
        if (
            not settings.admin_token
            or settings.admin_token == "replace-with-a-long-random-token"
            or (not hmac.compare_digest(token, settings.admin_token))
        ):
            raise HTTPException(401, "administrator credential required")

    def service():
        if app.state.service is None:
            raise HTTPException(503, "configured model failed to load; readiness is false")
        return app.state.service

    def report_index():
        root = settings.report_dir.resolve()
        result = {}
        for path in settings.report_dir.rglob("*.json"):
            if path.is_file() and path.resolve().is_relative_to(root):
                relative = path.relative_to(settings.report_dir).as_posix()
                result[relative.replace("/", "--").removesuffix(".json")] = path
        return result

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/api/v1/health")
    async def health():
        return {"status": "alive", "profile": settings.profile, "schema_version": SCHEMA_VERSION}

    @app.get("/api/v1/ready")
    async def ready():
        if app.state.service is None or not app.state.service.engine.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "profile": settings.profile,
                    "startup_error": app.state.startup_error,
                },
            )
        return {
            "ready": True,
            "profile": settings.profile,
            "generation_enabled": settings.profile == "local",
        }

    @app.get("/api/v1/models")
    async def models():
        return service().engine.metadata()

    @app.post("/api/v1/triage")
    async def triage(body: TriageRequest):
        if not body.message.strip():
            raise HTTPException(422, "message must contain text")
        result = await service().triage(body.message, body.mode)
        code = {
            "model_unavailable": 503,
            "overloaded": 429,
            "inference_failed": 500,
            "invalid_input": 422,
            "invalid_output": 502,
        }.get(result["status"], 200)
        return JSONResponse(status_code=code, content=result)

    @app.post("/api/v1/compare")
    async def compare(body: CompareRequest):
        # Sequential calls serialize shared adapter selection; all modes receive the exact same text.
        return {
            "results": [
                await service().triage(body.message, mode)
                for mode in ("zero_shot", "few_shot", "adapted")
            ],
            "fixture_notice": settings.profile == "fixture",
        }

    @app.get("/api/v1/dataset", dependencies=[Depends(admin)])
    async def dataset(split: Literal["train", "validation", "test"] = "train", offset: int = 0):
        if offset < 0:
            raise HTTPException(422, "offset must be nonnegative")
        rows = read_rows(settings.data_dir / f"{split}.jsonl")
        return {
            "split": split,
            "total": len(rows),
            "offset": offset,
            "rows": rows[offset : offset + 20],
            "manifest": json.loads((settings.data_dir / "manifest.json").read_text()),
        }

    @app.get("/api/v1/experiments", dependencies=[Depends(admin)])
    async def experiments():
        return {"reports": sorted(report_index())}

    @app.get("/api/v1/experiments/{report_id}", dependencies=[Depends(admin)])
    async def report(report_id: str):
        index = report_index()
        if report_id not in index:
            raise HTTPException(404, "unknown configured report")
        data = json.loads(index[report_id].read_text())

        # Raw generations and complete examples are available only in the authorized result route.
        def sanitize(value):
            if isinstance(value, dict):
                return {
                    k: sanitize(v)
                    for k, v in value.items()
                    if k
                    not in (
                        "raw",
                        "message",
                        "decoded_prompt",
                        "decoded_learned_target",
                        "failures",
                        "errors",
                        "requests_detail",
                        "response",
                        "gold",
                        "train_example_ids",
                        "fit_ids",
                    )
                }
            if isinstance(value, list):
                return [sanitize(v) for v in value]
            return value

        return sanitize(data)

    @app.get("/api/v1/experiments/{report_id}/results", dependencies=[Depends(admin)])
    async def results(report_id: str, offset: int = 0):
        index = report_index()
        if report_id not in index or offset < 0:
            raise HTTPException(404, "unknown report or offset")
        file = index[report_id].parent / "predictions.jsonl"
        if not file.exists() or not file.resolve().is_relative_to(settings.report_dir.resolve()):
            raise HTTPException(404, "per-example results unavailable")
        rows = read_rows(file)
        return {"total": len(rows), "rows": rows[offset : offset + 20]}

    @app.delete("/api/v1/experiments/{report_id}", dependencies=[Depends(admin)])
    async def delete_report(report_id: str):
        index = report_index()
        if report_id not in index:
            raise HTTPException(404, "unknown report")
        target = index[report_id]
        # Remove the report and associated stored output copies, not merely a UI entry.
        paths = [target]
        if target.name == "report.json":
            paths += [
                target.parent / name
                for name in ("predictions.jsonl", "runtime.json", "frozen-config.json")
            ]
        removed = []
        for path in paths:
            if path.exists() and path.resolve().is_relative_to(settings.report_dir.resolve()):
                path.unlink()
                removed.append(path.name)
        return {"deleted_files": removed}

    return app


app = create_app()
