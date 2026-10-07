"""Lightweight, dependency-free serving hardening.

Two small pieces of Starlette middleware that need no extra packages:

* :class:`RateLimitMiddleware` — a fixed-window per-client-IP request limit,
  enough to blunt accidental hammering or a naive scraper on a self-hosted box.
* :class:`AuthMiddleware` — optional ``Authorization: Bearer <token>`` gate on
  ``/api`` routes (health is always public). Disabled unless a token is set,
  so the zero-config local experience is unchanged.

Both are intentionally simple and in-process: this is a single-user, self-hosted
tool, not a multi-tenant service. They exist to make the app safe to expose on a
LAN, not to replace a real gateway.
"""

from __future__ import annotations

import hmac
import time
from collections import deque
from threading import Lock
from typing import Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-IP rate limit over ``/api`` routes."""

    def __init__(self, app, *, limit: int, window_seconds: int) -> None:
        super().__init__(app)
        self.limit = limit
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def _client_ip(self, request: Request) -> str:
        # Honor a single proxy hop; fall back to the socket peer.
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if self.limit <= 0 or not request.url.path.startswith("/api"):
            return await call_next(request)

        ip = self._client_ip(request)
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            dq = self._hits.setdefault(ip, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                retry = int(self.window - (now - dq[0])) + 1
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(max(1, retry))},
                )
            dq.append(now)
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """Optional bearer-token gate on ``/api`` routes (health stays public)."""

    def __init__(self, app, *, token: str) -> None:
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        needs_auth = (
            bool(self.token)
            and path.startswith("/api")
            and path != "/api/health"
            # CORS preflight carries no credentials by design; let the CORS
            # middleware answer it rather than 401-ing the preflight.
            and request.method != "OPTIONS"
        )
        if needs_auth:
            header = request.headers.get("authorization", "")
            provided = header[7:] if header.lower().startswith("bearer ") else ""
            if not provided or not hmac.compare_digest(provided, self.token):
                return JSONResponse(
                    {"detail": "unauthorized"}, status_code=401
                )
        return await call_next(request)
