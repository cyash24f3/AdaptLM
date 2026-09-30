import asyncio
import logging
import time

from fastapi.testclient import TestClient

from adaptlm.api.app import create_app
from adaptlm.config import Settings
from adaptlm.inference.engine import FixtureEngine, parse_output
from adaptlm.inference.service import InferenceService
from adaptlm.web.cloud import result_html


def test_fixture_journey_and_live_unavailable(client):
    meta = client.get("/api/v1/models").json()
    assert meta["profile"] == "fixture" and meta["base_model_id"] is None
    ready = client.get("/api/v1/ready").json()
    assert ready["ready"] and not ready["generation_enabled"]
    text = meta["demo_messages"][0]
    response = client.post("/api/v1/triage", json={"message": text, "mode": "fixture"})
    assert response.status_code == 200
    assert response.json()["raw_valid"]
    assert response.json()["usage"] is None
    assert "raw" not in response.json()
    response = client.post("/api/v1/triage", json={"message": text, "mode": "adapted"})
    assert response.status_code == 503 and response.json()["output"] is None
    comparison = client.post("/api/v1/compare", json={"message": text}).json()
    assert all(r["status"] == "model_unavailable" for r in comparison["results"])


def test_auth_and_error_redaction(client):
    assert client.get("/api/v1/dataset").status_code == 401
    assert client.get("/api/v1/experiments").status_code == 401
    headers = {"Authorization": "Bearer test-credential"}
    assert client.get("/api/v1/dataset", headers=headers).json()["total"] == 880
    secret = "private-secret@example.com"
    response = client.post("/api/v1/triage", json={"message": secret, "mode": "unknown"})
    assert response.status_code == 422 and secret not in response.text
    assert "request_id" in response.json()
    assert client.post("/api/v1/triage", json={"message": "x" * 100001}).status_code == 413


def test_ui_shell_and_protected_data(client):
    html = client.get("/").text
    assert all(f'id="{view}"' in html for view in ("triage", "compare", "dataset", "experiments"))
    assert "No paid APIs" in html
    assert client.get("/static/app.js").status_code == 200
    assert "frame-ancestors 'none'" in client.get("/").headers["Content-Security-Policy"]


def test_parse_malformed_oversized_and_fence_repair(settings):
    engine = FixtureEngine(settings)
    text = engine.metadata()["demo_messages"][0]
    raw = engine.generate(text, "fixture")["raw"]
    assert parse_output(raw, text)[1:3] == (True, 0)
    assert parse_output("```json\n" + raw + "\n```", text)[1:3] == (False, 1)
    assert parse_output("{broken", text)[0] is None
    assert parse_output(raw * 1000, text, 1000)[0] is None


class SlowEngine(FixtureEngine):
    def generate(self, message, mode):
        time.sleep(0.12)
        return super().generate(message, mode)


def test_queue_bounds_and_cancellation_holds_capacity(settings):
    async def scenario():
        settings.queue_capacity = 0
        engine = SlowEngine(settings)
        service = InferenceService(settings, engine)
        message = engine.metadata()["demo_messages"][0]
        task = asyncio.create_task(service.triage(message, "fixture"))
        await asyncio.sleep(0.03)
        overloaded = await service.triage(message, "fixture")
        assert overloaded["status"] == "overloaded"
        task.cancel()
        task.cancel()
        await asyncio.sleep(0.01)
        assert service.admitted == 1
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert service.admitted == 1
        assert (await service.triage(message, "fixture"))["status"] == "overloaded"
        await asyncio.sleep(0.13)
        assert service.admitted == 0
        assert (await service.triage(message, "fixture"))["final_valid"]

    asyncio.run(scenario())


def test_queue_timeout_and_log_redaction(settings, caplog):
    async def scenario():
        settings.queue_timeout_seconds = 0.01
        engine = SlowEngine(settings)
        service = InferenceService(settings, engine)
        message = engine.metadata()["demo_messages"][0]
        first = asyncio.create_task(service.triage(message, "fixture"))
        await asyncio.sleep(0.01)
        second = await service.triage(message, "fixture")
        assert second["status"] == "overloaded"
        await first
        assert message not in caplog.text
        assert "request_id=" in caplog.text

    caplog.set_level(logging.INFO, logger="adaptlm.requests")
    asyncio.run(scenario())


def test_truncation_is_never_valid(settings):
    class Truncated(FixtureEngine):
        def generate(self, message, mode):
            return dict(super().generate(message, mode), truncated=True)

    engine = Truncated(settings)
    response = asyncio.run(
        InferenceService(settings, engine).triage(engine.metadata()["demo_messages"][0], "fixture")
    )
    assert not response["raw_valid"] and response["status"] == "invalid_output"


def test_public_cloud_result_escapes_untrusted_text(settings):
    engine = FixtureEngine(settings)
    response = asyncio.run(
        InferenceService(settings, engine).triage(engine.metadata()["demo_messages"][0], "fixture")
    )
    response["output"]["summary_claims"][0]["text"] = '<script>alert("input")</script>'
    rendered = result_html(response)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_report_deletion_removes_stored_outputs(tmp_path):
    root = tmp_path / "reports" / "experiment"
    root.mkdir(parents=True)
    (root / "report.json").write_text('{"metrics":{},"message":"private","raw":"secret"}')
    (root / "predictions.jsonl").write_text('{"message":"private"}\n')
    settings = Settings(
        profile="fixture", report_dir=tmp_path / "reports", admin_token="test", _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer test"}
        response = client.get("/api/v1/experiments/experiment--report", headers=headers)
        assert "secret" not in response.text and "private" not in response.text
        assert (
            client.delete("/api/v1/experiments/experiment--report", headers=headers).status_code
            == 200
        )
    assert not (root / "report.json").exists() and not (root / "predictions.jsonl").exists()
