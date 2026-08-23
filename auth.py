# auth.py

import hashlib
import secrets

from fastapi import Request, HTTPException


HOD_USERNAME = "hod"

HOD_PASSWORD_HASH = hashlib.sha256(
    "changeme123".encode()
).hexdigest()

active_sessions = {}


def verify_login(username, password):
    return (
        username == HOD_USERNAME
        and hashlib.sha256(
            password.encode()
        ).hexdigest()
        == HOD_PASSWORD_HASH
    )


def create_session(username):
    token = secrets.token_hex(32)
    active_sessions[token] = username
    return token


def require_login(request: Request):
    token = request.cookies.get(
        "session_token"
    )

    if (
        not token
        or token not in active_sessions
    ):
        raise HTTPException(
            status_code=303,
            headers={
                "Location": "/login"
            },
        )

    return active_sessions[token]