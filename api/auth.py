"""Authentication helpers for the Echo Chamber Detector REST API.

Supports two authentication modes:

1. **JWT Bearer tokens** — ``Authorization: Bearer <jwt>``
   JWT is decoded using HMAC-SHA256 with ``JWT_SECRET_KEY`` from the
   environment (default ``dev-secret-key`` for development/testing).
   The ``sub`` claim is used as the user ID.

2. **API key** — ``Authorization: Bearer <api-key>``
   The token is checked against ``API_KEYS`` env var (comma-separated list).
   The API key string itself is returned as the ``userId``.

If neither check passes, HTTP 401 is raised.

References: Requirements 7.8, 10.2
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: HMAC secret key for JWT signing/verification.  Override in production.
JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "dev-secret-key")

#: Comma-separated list of valid API keys.  Empty string disables API-key auth.
_RAW_API_KEYS: str = os.environ.get("API_KEYS", "")
VALID_API_KEYS: frozenset[str] = frozenset(
    k.strip() for k in _RAW_API_KEYS.split(",") if k.strip()
)

# ---------------------------------------------------------------------------
# FastAPI HTTP Bearer scheme
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Minimal JWT encode/decode (HMAC-SHA256, HS256)
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    """Base64-URL encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64-URL decode, re-adding padding as needed."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def encode_jwt(payload: dict, secret: str = JWT_SECRET_KEY) -> str:
    """Encode a JWT token using HS256 (HMAC-SHA256).

    Args:
        payload: Claims dict; must contain at least ``"sub"``.
        secret:  HMAC secret key.

    Returns:
        Compact JWT string ``header.payload.signature``.
    """
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url_encode(json.dumps(payload).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url_encode(sig)}"


def decode_jwt(token: str, secret: str = JWT_SECRET_KEY) -> dict:
    """Decode and verify a JWT token (HS256 only).

    Args:
        token:  Compact JWT string.
        secret: HMAC secret key.

    Returns:
        Payload dict extracted from the token.

    Raises:
        ValueError: If the token is malformed or the signature is invalid.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format: expected 3 dot-separated segments")

    header_b64, payload_b64, sig_b64 = parts

    # Verify signature
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    provided_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("JWT signature verification failed")

    # Decode payload
    payload_bytes = _b64url_decode(payload_b64)
    return json.loads(payload_bytes)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
) -> str:
    """FastAPI dependency that returns the authenticated user's ID.

    Tries JWT verification first; falls back to API-key lookup.

    Args:
        credentials: HTTP Bearer credentials from the ``Authorization`` header.

    Returns:
        ``userId`` string — extracted from the JWT ``sub`` claim, or the
        API key string itself when API-key auth succeeds.

    Raises:
        HTTPException(401): If no credentials are provided or both checks fail.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # --- Attempt 1: JWT verification ---
    try:
        payload = decode_jwt(token, JWT_SECRET_KEY)
        sub = payload.get("sub")
        if sub:
            return str(sub)
        raise ValueError("JWT payload missing 'sub' claim")
    except ValueError as exc:
        logger.debug("JWT verification failed: %s", exc)

    # --- Attempt 2: API key lookup ---
    if VALID_API_KEYS and token in VALID_API_KEYS:
        return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
