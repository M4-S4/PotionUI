from typing import Optional
from pydantic import BaseModel

from src.platform.security.email_address import EmailAddress


class UserCreate(BaseModel):
    username: str
    email: EmailAddress
    password: str
    account_type: Optional[str] = "USER"


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailAddress] = None
    password: Optional[str] = None
    account_type: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    account_type: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_login: Optional[str] = None
    avatar_url: Optional[str] = None
