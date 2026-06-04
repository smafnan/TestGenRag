"""
Optional authentication via JWT bearer tokens.

Disabled by default (AUTH_ENABLED unset/false) so local development and demos
need no login. When enabled, it validates a bearer token against a JWKS
endpoint, supporting both:

    - AWS Cognito  (set COGNITO_REGION + COGNITO_USER_POOL_ID)
    - Okta / SAML-backed OIDC, or any OIDC IdP (set OIDC_ISSUER + OIDC_AUDIENCE)

Okta federates SAML enterprise logins and issues standard OIDC JWTs, so the
same verification path covers "Okta/SAML" and generic OIDC providers.

Use as a FastAPI dependency:  user = Depends(require_user)
"""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "false").lower() == "true"


def _issuer_and_audience() -> tuple[str, Optional[str]]:
    region = os.getenv("COGNITO_REGION")
    pool = os.getenv("COGNITO_USER_POOL_ID")
    if region and pool:
        issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool}"
        return issuer, os.getenv("COGNITO_APP_CLIENT_ID")
    # Generic OIDC / Okta
    return os.getenv("OIDC_ISSUER", ""), os.getenv("OIDC_AUDIENCE")


def _verify(token: str) -> dict:
    """Validate signature + claims against the IdP's JWKS. Raises on failure."""
    from jose import jwt  # python-jose
    import urllib.request
    import json as _json

    issuer, audience = _issuer_and_audience()
    if not issuer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_ENABLED but no issuer configured.",
        )

    jwks_uri = issuer.rstrip("/") + "/.well-known/jwks.json"
    if "cognito-idp" in issuer:
        jwks_uri = issuer.rstrip("/") + "/.well-known/jwks.json"

    with urllib.request.urlopen(jwks_uri, timeout=5) as resp:
        jwks = _json.loads(resp.read())

    headers = jwt.get_unverified_header(token)
    key = next((k for k in jwks["keys"] if k["kid"] == headers["kid"]), None)
    if key is None:
        raise HTTPException(status_code=401, detail="Signing key not found.")

    options = {"verify_aud": bool(audience)}
    try:
        return jwt.decode(
            token,
            key,
            algorithms=[headers.get("alg", "RS256")],
            audience=audience,
            issuer=issuer,
            options=options,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


def require_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency. Returns the token claims (or an anonymous principal
    when auth is disabled).
    """
    if not auth_enabled():
        return {"sub": "anonymous", "auth": "disabled"}
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return _verify(creds.credentials)
