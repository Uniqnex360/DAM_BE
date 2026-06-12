from pydantic import BaseModel
from typing import Optional,Any


class Token(BaseModel):
    access_token: str
    token_type: str
    impersonated_user: Optional[Any] = None
class TokenPayload(BaseModel):
    sub: Optional[str] = None