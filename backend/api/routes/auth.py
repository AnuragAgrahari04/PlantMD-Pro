"""
Authentication Routes — JWT-based auth
For production: replace hardcoded users with a PostgreSQL user table.
"""
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from core.config import settings

logger = structlog.get_logger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def _hash_password(password: str) -> str:
    """SHA-256 based password hashing — no external bcrypt dependency."""
    salt = "plantmd_salt_v2"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _verify_password(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_password(plain), hashed)


# ── Fake user DB (replace with PostgreSQL in production) ─────────────────────
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@plantmd.io",
        "hashed_password": _hash_password("changeme123"),
        "role": "admin",
        "is_active": True,
    },
    "demo": {
        "username": "demo",
        "email": "demo@plantmd.io",
        "hashed_password": _hash_password("demo"),
        "role": "user",
        "is_active": True,
    },
}


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class User(BaseModel):
    username: str
    email: str
    role: str
    is_active: bool


def verify_password(plain: str, hashed: str) -> bool:
    return _verify_password(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        raise credentials_exception

    user = FAKE_USERS_DB.get(token_data.username)
    if user is None or not user["is_active"]:
        raise credentials_exception
    return User(**user)


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue JWT token for valid credentials."""
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        logger.warning("Failed login attempt", username=form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    logger.info("User logged in", username=user["username"])
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user profile."""
    return current_user