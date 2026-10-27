"""CPU ASGI/utility contract tests; no vLLM import, worker, IPC, or GPU.

The router runs against a mock EngineClient. Existing authentication and the
AsyncLLM method are loaded from their actual ASTs to avoid GPU dependencies.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import secrets
import sys
import types
import unittest
from unittest.mock import patch

try:
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from starlette.datastructures import Headers, URL
except ImportError:
    httpx = None


ROOT = Path(__file__).resolve().parents[1] / "vllm-source" / "vllm"


def load_router():
    logger_stub = types.ModuleType("vllm.logger")
    logger_stub.init_logger = logging.getLogger
    path = ROOT / "entrypoints/serve/ppd/api_router.py"
    spec = importlib.util.spec_from_file_location("ppd_observer_router_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"vllm.logger": logger_stub}):
        spec.loader.exec_module(module)
    return module


def load_ast_definition(path, name, namespace, parent=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if parent:
        nodes = next(node for node in nodes if getattr(node, "name", None) == parent).body
    node = next(node for node in nodes if getattr(node, "name", None) == name)
    isolated = ast.Module(body=[node], type_ignores=[])
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace[name]


class FakeClient:
    def __init__(self):
        self.model_config = types.SimpleNamespace(
            max_model_len=16, get_vocab_size=lambda: 64
        )
        self.calls = []
        self.started = asyncio.Event()
        self.release = None
        self.error = None

    async def observe_ppd_state(self, payload):
        self.calls.append(payload)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error:
            raise self.error
        return {"status": "ok", "source": "mock_engine"}


@unittest.skipIf(httpx is None, "FastAPI/httpx not installed; no packages installed")
class ObserverRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.module = load_router()
        self.client = FakeClient()
        self.app = FastAPI()
        self.app.state.args = types.SimpleNamespace(api_key=["unit-test-key"])
        self.app.state.engine_client = self.client
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "1"}):
            self.module.attach_router(self.app)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://observer"
        )

    async def asyncTearDown(self):
        if self.client.release is not None:
            self.client.release.set()
        task = self.app.state.ppd_observer_gate.task
        if task is not None:
            try:
                await task
            except Exception:
                pass
        await self.http.aclose()

    async def post(self, payload):
        return await self.http.post("/v1/ppd/observe", json=payload)

    async def test_disabled_flag_does_not_register_route(self):
        for value in ("", "0", "true"):
            app = FastAPI()
            with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": value}):
                self.module.attach_router(app)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://observer"
            ) as client:
                self.assertEqual((await client.post("/v1/ppd/observe", json={})).status_code, 404)

    async def test_state_only_no_probe(self):
        response = await self.post({})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.calls, [{}])

    async def test_supplied_ids_forward_without_rendering(self):
        payload = {
            "prompt_token_ids": [1, 2], "cache_salt": "isolation-salt",
            "expected_node_epoch": "epoch-1", "request_ids": ["request-1"],
        }
        response = await self.post(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.calls, [payload])
        self.assertNotIn("prompt_token_ids", response.json())

    async def test_invalid_payloads_do_not_reach_engine(self):
        bad_payloads = [
            [], None, {"messages": []}, {"prompt_token_ids": []},
            {"prompt_token_ids": [True]}, {"prompt_token_ids": [-1]},
            {"prompt_token_ids": [64]}, {"prompt_token_ids": [1.5]},
            {"cache_salt": "salt"}, {"expected_node_epoch": ""},
            {"expected_node_epoch": "a" * 257}, {"request_ids": ["x"] * 129},
            {"request_ids": ["x", "x"]}, {"request_ids": [1]},
        ]
        with patch.object(self.module, "_MIN_INTERVAL_S", 0):
            for payload in bad_payloads:
                with self.subTest(payload=payload):
                    self.assertEqual((await self.post(payload)).status_code, 400)
        self.assertEqual(self.client.calls, [])

    async def test_model_token_cap(self):
        self.assertEqual((await self.post({"prompt_token_ids": [1] * 17})).status_code, 413)
        self.assertEqual(self.client.calls, [])

    async def test_global_token_cap(self):
        self.client.model_config.max_model_len = 1000000
        self.assertEqual((await self.post({"prompt_token_ids": [1] * 262145})).status_code, 413)
        self.assertEqual(self.client.calls, [])

    async def test_declared_body_cap(self):
        response = await self.http.post(
            "/v1/ppd/observe", content=b"{}", headers={"content-length": "2097153"}
        )
        self.assertEqual(response.status_code, 413)

    async def test_stream_body_cap_without_declared_length(self):
        async def chunks():
            yield b" " * (1024 * 1024)
            yield b" " * (1024 * 1024 + 1)
        response = await self.http.post("/v1/ppd/observe", content=chunks())
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.calls, [])

    async def test_body_timeout_releases_non_dispatched_slot(self):
        async def stalled():
            await asyncio.Event().wait()
            yield b"{}"
        with patch.object(self.module, "_TIMEOUT_S", 0.01):
            response = await self.http.post("/v1/ppd/observe", content=stalled())
        self.assertEqual(response.status_code, 504)
        self.assertFalse(self.app.state.ppd_observer_gate.busy)
        self.assertEqual(self.client.calls, [])

    async def test_minimum_interval(self):
        with patch.object(self.module, "_MIN_INTERVAL_S", 100):
            self.assertEqual((await self.post({})).status_code, 200)
            self.assertEqual((await self.post({})).status_code, 429)

    async def test_timeout_keeps_flight_until_engine_finishes(self):
        self.client.release = asyncio.Event()
        with patch.object(self.module, "_TIMEOUT_S", 0.01):
            self.assertEqual((await self.post({})).status_code, 504)
        gate = self.app.state.ppd_observer_gate
        self.assertTrue(gate.busy)
        self.assertFalse(gate.task.done())
        with patch.object(self.module, "_MIN_INTERVAL_S", 0):
            self.assertEqual((await self.post({})).status_code, 429)
        task = gate.task
        self.client.release.set()
        await task
        await asyncio.sleep(0)
        self.assertFalse(gate.busy)
        self.assertEqual(len(self.client.calls), 1)

    async def test_cancellation_keeps_flight_until_engine_finishes(self):
        self.client.release = asyncio.Event()
        waiter = asyncio.create_task(self.post({}))
        await self.client.started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        gate = self.app.state.ppd_observer_gate
        self.assertTrue(gate.busy)
        self.assertFalse(gate.task.cancelled())
        with patch.object(self.module, "_MIN_INTERVAL_S", 0):
            self.assertEqual((await self.post({})).status_code, 429)

    async def test_engine_errors_are_sanitized(self):
        with patch.object(self.module, "_MIN_INTERVAL_S", 0):
            for error, status in (
                (NotImplementedError("sensitive-id"), 501),
                (ValueError("sensitive-id"), 400),
                (RuntimeError("sensitive-id"), 503),
            ):
                self.client.error = error
                response = await self.post({})
                self.assertEqual(response.status_code, status)
                self.assertNotIn("sensitive-id", response.text)

    async def test_actual_v1_auth_middleware_applies(self):
        middleware = load_ast_definition(
            ROOT / "entrypoints/openai/api_server.py", "AuthenticationMiddleware",
            {"hashlib": hashlib, "secrets": secrets, "URL": URL,
             "Headers": Headers, "JSONResponse": JSONResponse},
        )
        self.app.add_middleware(middleware, tokens=["unit-test-key"])
        denied = await self.post({})
        self.assertEqual(denied.status_code, 401)
        allowed = await self.http.post(
            "/v1/ppd/observe", json={},
            headers={"Authorization": "Bearer unit-test-key"},
        )
        self.assertEqual(allowed.status_code, 200)


class ObserverEngineDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        class AsyncMPClient:
            def __init__(self):
                self.calls = []

            async def call_utility_async(self, *args):
                self.calls.append(args)
                return {"ok": True}

        class DPAsyncMPClient(AsyncMPClient):
            pass

        self.async_type = AsyncMPClient
        self.dp_type = DPAsyncMPClient
        method = load_ast_definition(
            ROOT / "v1/engine/async_llm.py", "observe_ppd_state",
            {"os": os, "AsyncMPClient": AsyncMPClient, "DPAsyncMPClient": DPAsyncMPClient},
            parent="AsyncLLM",
        )
        self.engine = types.SimpleNamespace(
            engine_core=AsyncMPClient(),
            vllm_config=types.SimpleNamespace(
                parallel_config=types.SimpleNamespace(data_parallel_size=1)
            ),
        )
        self.invoke = types.MethodType(method, self.engine)

    async def test_single_engine_utility_only(self):
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "1"}):
            self.assertEqual(await self.invoke({}), {"ok": True})
        self.assertEqual(self.engine.engine_core.calls, [("observe_ppd_state", {})])

    async def test_disabled(self):
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "0"}):
            with self.assertRaises(NotImplementedError):
                await self.invoke({})

    async def test_dp_size_rejected(self):
        self.engine.vllm_config.parallel_config.data_parallel_size = 2
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "1"}):
            with self.assertRaises(NotImplementedError):
                await self.invoke({})
        self.assertEqual(self.engine.engine_core.calls, [])

    async def test_dp_client_rejected_even_at_size_one(self):
        self.engine.engine_core = self.dp_type()
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "1"}):
            with self.assertRaises(NotImplementedError):
                await self.invoke({})
        self.assertEqual(self.engine.engine_core.calls, [])

    async def test_inprocess_client_rejected(self):
        self.engine.engine_core = object()
        with patch.dict(os.environ, {"VLLM_PPD_OBSERVER_ENABLED": "1"}):
            with self.assertRaises(NotImplementedError):
                await self.invoke({})


if __name__ == "__main__":
    unittest.main()
