import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings


@dataclass(frozen=True)
class Principal:
    user_id: str


bearer = HTTPBearer(auto_error=False)


def get_principal(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> Principal:
    tokens = get_settings().auth_tokens
    if (not tokens or any(not user or len(user) > 255 or len(token) < 32 for user, token in tokens.items())
            or len(set(tokens.values())) != len(tokens)):
        raise HTTPException(503, "User authentication is not configured")
    if credentials and credentials.scheme.lower() == "bearer":
        for user, token in tokens.items():
            if secrets.compare_digest(credentials.credentials.encode(), token.encode()):
                return Principal(user)
    raise HTTPException(401, "Valid bearer token required", headers={"WWW-Authenticate": "Bearer"})


PrincipalDep = Annotated[Principal, Depends(get_principal)]
