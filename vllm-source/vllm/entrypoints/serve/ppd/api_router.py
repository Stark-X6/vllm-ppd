# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded, owner-thread PPD observations; never enqueue generation."""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()

_MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_PROBE_TOKENS = 262144
_MAX_REQUEST_IDS = 128
_MAX_STRING_LENGTH = 256
_MIN_INTERVAL_S = 0.1
_TIMEOUT_S = 5.0
_ALLOWED_FIELDS = {
    "prompt_token_ids",
    "cache_salt",
    "expected_node_epoch",
    "request_ids",
}


class _ObserverGate:
    """One API-process flight, including time spent reading the bounded body.

    A timed-out/cancelled HTTP waiter does not cancel the EngineCore utility.
    Retain its slot until completion, so repeated timeouts cannot flood IPC.
    This does not replace any EngineCore-wide limits across API processes.
    """

    def __init__(self) -> None:
        self.busy = False
        self.last_started = float("-inf")
        self.task: asyncio.Task | None = None

    def reserve(self) -> None:
        now = time.monotonic()
        if self.busy or now - self.last_started < _MIN_INTERVAL_S:
            raise HTTPException(429, "PPD observer busy or rate limited")
        self.busy = True
        self.last_started = now

    def completed(self, task: asyncio.Task) -> None:
        # Retrieve exceptions even if the HTTP waiter timed out or disconnected.
        if not task.cancelled():
            task.exception()
        if self.task is task:
            self.task = None
            self.busy = False


async def _read_payload(request: Request) -> dict[str, Any]:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        if size < 0:
            raise HTTPException(400, "Invalid Content-Length")
        if size > _MAX_BODY_BYTES:
            raise HTTPException(413, "PPD observer body exceeds limit")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_BODY_BYTES:
            raise HTTPException(413, "PPD observer body exceeds limit")
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "PPD observer requires a JSON object") from None
    if not isinstance(payload, dict) or set(payload) - _ALLOWED_FIELDS:
        raise HTTPException(400, "Invalid PPD observer fields")
    return payload


def _validate_payload(payload: dict[str, Any], engine_client: Any) -> None:
    for key in ("cache_salt", "expected_node_epoch"):
        if key in payload and (
            not isinstance(payload[key], str)
            or not 1 <= len(payload[key]) <= _MAX_STRING_LENGTH
        ):
            raise HTTPException(400, f"Invalid {key}")
    if "cache_salt" in payload and "prompt_token_ids" not in payload:
        raise HTTPException(400, "cache_salt requires a token probe")
    if "request_ids" in payload:
        request_ids = payload["request_ids"]
        if not isinstance(request_ids, list) or len(request_ids) > _MAX_REQUEST_IDS:
            raise HTTPException(400, "Invalid request_ids")
        if any(
            not isinstance(value, str)
            or not 1 <= len(value) <= _MAX_STRING_LENGTH
            for value in request_ids
        ):
            raise HTTPException(400, "Invalid request_ids")
        if len(set(request_ids)) != len(request_ids):
            raise HTTPException(400, "Duplicate request_ids")
    if "prompt_token_ids" not in payload:
        return
    token_ids = payload["prompt_token_ids"]
    if not isinstance(token_ids, list) or not token_ids:
        raise HTTPException(400, "prompt_token_ids must be a nonempty list")
    model_config = engine_client.model_config
    token_cap = min(_MAX_PROBE_TOKENS, model_config.max_model_len)
    if len(token_ids) > token_cap:
        raise HTTPException(413, "PPD observer token probe exceeds limit")
    vocab_size = model_config.get_vocab_size()
    if any(
        type(value) is not int or not 0 <= value < vocab_size
        for value in token_ids
    ):
        raise HTTPException(400, "Invalid prompt_token_ids")


@router.post("/v1/ppd/observe")
async def observe_ppd_state(request: Request) -> JSONResponse:
    """State-only {} or caller-supplied token IDs, not a chat renderer.

    The endpoint inherits /v1 API-key authentication when configured. It does
    not attest that supplied IDs match a later chat request's serialization.
    """
    gate: _ObserverGate = request.app.state.ppd_observer_gate
    gate.reserve()
    dispatched = False
    try:
        try:
            payload = await asyncio.wait_for(_read_payload(request), _TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(504, "PPD observer body read timed out") from None
        client = request.app.state.engine_client
        _validate_payload(payload, client)
        task = asyncio.create_task(client.observe_ppd_state(payload))
        gate.task = task
        task.add_done_callback(gate.completed)
        dispatched = True
        try:
            result = await asyncio.wait_for(asyncio.shield(task), _TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(
                504, "PPD observer timed out; work remains bounded"
            ) from None
        except NotImplementedError:
            raise HTTPException(501, "Unsupported PPD observer engine layout") from None
        except ValueError:
            raise HTTPException(400, "Invalid PPD observer request") from None
        except Exception:
            # Engine exception messages may contain probe/request identifiers.
            raise HTTPException(503, "PPD observer unavailable") from None
        return JSONResponse(result)
    finally:
        if not dispatched:
            gate.busy = False


def attach_router(app: FastAPI) -> None:
    if os.environ.get("VLLM_PPD_OBSERVER_ENABLED") != "1":
        return
    args = getattr(app.state, "args", None)
    api_keys = getattr(args, "api_key", None) or [os.environ.get("VLLM_API_KEY")]
    if not any(api_keys):
        logger.warning(
            "PPD observer opted in without an API key; restrict network access. "
            "This endpoint exposes request/cache state."
        )
    app.state.ppd_observer_gate = _ObserverGate()
    app.include_router(router)
