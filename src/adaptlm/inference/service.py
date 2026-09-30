import asyncio
import logging
import time
from uuid import uuid4

from adaptlm.config import Settings
from adaptlm.inference.engine import FixtureEngine, TransformersEngine, parse_output

logger = logging.getLogger("adaptlm.requests")


class InferenceService:
    """One serialized worker: adapter toggles and generation cannot overlap."""

    def __init__(self, settings: Settings, engine=None):
        self.settings = settings
        self.engine = engine or (
            FixtureEngine(settings)
            if settings.profile == "fixture"
            else TransformersEngine(settings)
        )
        self.semaphore = asyncio.Semaphore(1)
        self.admitted = 0

    def failure(self, request_id, mode, status, detail):
        return {
            "request_id": request_id,
            "status": status,
            "mode": mode,
            "output": None,
            "metadata": self.engine.metadata(),
            "raw_valid": False,
            "final_valid": False,
            "repair_count": 0,
            "usage": None,
            "timings": {},
            "error": detail,
        }

    async def _generate(self, message, mode):
        # The worker owns capacity even if its HTTP caller is cancelled repeatedly.
        try:
            return await asyncio.to_thread(self.engine.generate, message, mode)
        finally:
            self.semaphore.release()
            self.admitted -= 1

    @staticmethod
    def _observe_worker(task):
        # A disconnected caller cannot retrieve an eventual exception.
        if not task.cancelled():
            task.exception()

    async def triage(self, message, mode, include_raw=False):
        request_id = str(uuid4())
        if mode not in self.engine.metadata()["available_modes"]:
            return self.failure(
                request_id, mode, "model_unavailable", "configured model mode unavailable"
            )
        if len(message) > self.settings.max_chars:
            return self.failure(
                request_id, mode, "invalid_input", "input character budget exceeded"
            )
        # No await between admission test and increment: atomic within one event loop.
        if self.admitted >= 1 + self.settings.queue_capacity:
            return self.failure(request_id, mode, "overloaded", "inference queue is full")
        self.admitted += 1
        start = time.perf_counter()
        acquired = False
        worker = None
        try:
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(), timeout=self.settings.queue_timeout_seconds
                )
                acquired = True
            except TimeoutError:
                return self.failure(request_id, mode, "overloaded", "queue admission timed out")
            queue_wait = time.perf_counter() - start
            worker = asyncio.create_task(self._generate(message, mode))
            worker.add_done_callback(self._observe_worker)
            generated = await asyncio.shield(worker)
            validated_at = time.perf_counter()
            record, raw_valid, repairs, error = parse_output(
                generated["raw"], message, self.settings.max_output_chars
            )
            if generated["truncated"]:
                record, raw_valid, error = (
                    None,
                    False,
                    "generation reached token limit without stop token",
                )
            response = {
                "request_id": request_id,
                "status": ("ok" if record.triage_status == "triaged" else record.triage_status)
                if record
                else "invalid_output",
                "mode": mode,
                "output": record.model_dump(mode="json") if record else None,
                "metadata": self.engine.metadata(),
                "raw_valid": raw_valid,
                "repair_count": repairs,
                "final_valid": record is not None,
                "error": error,
                "usage": generated["usage"],
                "truncated": generated["truncated"],
                "process_rss_bytes": generated["process_rss_bytes"],
                "accelerator_allocated_bytes": generated["accelerator_allocated_bytes"],
                "timings": generated["timings"]
                | {
                    "queue_wait_seconds": queue_wait,
                    "validation_seconds": time.perf_counter() - validated_at,
                    "total_seconds": time.perf_counter() - start,
                },
            }
            if include_raw:
                response["raw"] = generated["raw"]
            logger.info(
                "request_id=%s mode=%s status=%s queue_depth=%s seconds=%.3f",
                request_id,
                mode,
                response["status"],
                self.admitted,
                response["timings"]["total_seconds"],
            )
            return response
        except LookupError:
            return self.failure(
                request_id, mode, "model_unavailable", "no configured output for this input"
            )
        except ValueError:
            return self.failure(request_id, mode, "invalid_input", "token or input budget exceeded")
        except Exception:
            logger.error("request_id=%s mode=%s status=inference_failed", request_id, mode)
            return self.failure(
                request_id, mode, "inference_failed", "generation failed; inspect local runtime"
            )
        finally:
            if worker is None:
                if acquired:
                    self.semaphore.release()
                self.admitted -= 1
