"""OpenAI Codex OAuth authentication.

Uses the official OAuth 2.0 Authorization Code + PKCE flow from the OpenAI Codex CLI.
The ChatGPT subscription token must be used with the ChatGPT backend Responses API
at chatgpt.com/backend-api/codex/responses.
"""

import hashlib
import base64
import secrets
import json
import time
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs, quote
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import requests as http_requests

# OAuth constants from the official OpenAI Codex CLI source
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/oauth/authorize"
TOKEN_URL = f"{ISSUER}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPES = "openid profile email offline_access"

# ChatGPT backend (where subscription tokens work)
CHATGPT_BACKEND_URL = "https://chatgpt.com/backend-api"

# Where to cache tokens
TOKEN_CACHE_DIR = Path.home() / ".tradingagents"
TOKEN_CACHE_FILE = TOKEN_CACHE_DIR / "codex_tokens.json"


@dataclass
class CodexAuthResult:
    """Result from Codex OAuth login."""
    access_token: str
    account_id: str = ""
    base_url: str = CHATGPT_BACKEND_URL
    extra_headers: dict = field(default_factory=dict)
    has_api_key: bool = False


def _decode_jwt_payload(token: str) -> dict:
    """Decode a JWT payload without verification."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    payload_b64 = parts[1]
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes)
    except Exception:
        return {}


def _extract_account_id(token: str) -> str:
    """Extract chatgpt_account_id from a JWT's auth claims."""
    payload = _decode_jwt_payload(token)
    auth_claims = payload.get("https://api.openai.com/auth", {})
    return auth_claims.get("chatgpt_account_id", "")


class CodexOAuth:
    """OpenAI Codex OAuth 2.0 + PKCE authentication handler."""

    def __init__(self):
        self.code_verifier = None
        self.code_challenge = None
        self.state = None
        self.tokens = None

    def _generate_pkce(self):
        verifier_bytes = secrets.token_bytes(64)
        self.code_verifier = (
            base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")
        )
        digest = hashlib.sha256(self.code_verifier.encode("ascii")).digest()
        self.code_challenge = (
            base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        )

    def _generate_state(self):
        self.state = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .rstrip(b"=")
            .decode("ascii")
        )

    def _get_authorization_url(self) -> str:
        self._generate_pkce()
        self._generate_state()
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": self.state,
            "originator": "codex_cli_rs",
        }
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    def _post_token(self, body_str: str) -> dict:
        resp = http_requests.post(
            TOKEN_URL,
            data=body_str.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if not resp.ok:
            try:
                error_detail = resp.json()
            except Exception:
                error_detail = resp.text[:500]
            raise RuntimeError(
                f"Token request failed ({resp.status_code}): {error_detail}"
            )
        return resp.json()

    def _exchange_code(self, authorization_code: str):
        body = (
            f"grant_type=authorization_code"
            f"&code={quote(authorization_code, safe='')}"
            f"&redirect_uri={quote(REDIRECT_URI, safe='')}"
            f"&client_id={quote(CLIENT_ID, safe='')}"
            f"&code_verifier={quote(self.code_verifier, safe='')}"
        )
        data = self._post_token(body)
        self.tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "id_token": data.get("id_token", ""),
            "expires_at": time.time() + data.get("expires_in", 3600),
        }

    def _try_token_exchange(self) -> Optional[str]:
        if not self.tokens.get("id_token"):
            return None
        try:
            body = (
                f"grant_type={quote('urn:ietf:params:oauth:grant-type:token-exchange', safe='')}"
                f"&client_id={quote(CLIENT_ID, safe='')}"
                f"&requested_token=openai-api-key"
                f"&subject_token={quote(self.tokens['id_token'], safe='')}"
                f"&subject_token_type={quote('urn:ietf:params:oauth:token-type:id_token', safe='')}"
            )
            data = self._post_token(body)
            return data.get("access_token")
        except RuntimeError:
            return None

    def _build_result(self) -> CodexAuthResult:
        api_key = self._try_token_exchange()
        if api_key:
            return CodexAuthResult(
                access_token=api_key,
                base_url="https://api.openai.com/v1",
                has_api_key=True,
            )

        access_token = self.tokens["access_token"]
        account_id = _extract_account_id(access_token)
        if not account_id and self.tokens.get("id_token"):
            account_id = _extract_account_id(self.tokens["id_token"])

        return CodexAuthResult(
            access_token=access_token,
            account_id=account_id,
            base_url=CHATGPT_BACKEND_URL,
            extra_headers={
                "chatgpt-account-id": account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "codex_cli_rs",
            },
            has_api_key=False,
        )

    def _refresh_tokens(self):
        body = (
            f"grant_type=refresh_token"
            f"&refresh_token={quote(self.tokens['refresh_token'], safe='')}"
            f"&client_id={quote(CLIENT_ID, safe='')}"
        )
        data = self._post_token(body)
        self.tokens["access_token"] = data["access_token"]
        self.tokens["refresh_token"] = data["refresh_token"]
        if "id_token" in data:
            self.tokens["id_token"] = data["id_token"]
        self.tokens["expires_at"] = time.time() + data.get("expires_in", 3600)

    def _save_tokens(self):
        TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE_FILE.write_text(json.dumps(self.tokens, indent=2))
        TOKEN_CACHE_FILE.chmod(0o600)

    def _load_tokens(self) -> bool:
        if TOKEN_CACHE_FILE.exists():
            try:
                self.tokens = json.loads(TOKEN_CACHE_FILE.read_text())
                return bool(self.tokens.get("refresh_token"))
            except (json.JSONDecodeError, KeyError):
                return False
        return False

    def login(self) -> CodexAuthResult:
        """Run the full browser-based OAuth login flow."""
        if TOKEN_CACHE_FILE.exists():
            TOKEN_CACHE_FILE.unlink()

        auth_url = self._get_authorization_url()
        auth_code_holder = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = parse_qs(parsed.query)
                auth_code_holder["code"] = params.get("code", [None])[0]
                auth_code_holder["state"] = params.get("state", [None])[0]
                auth_code_holder["error"] = params.get("error", [None])[0]
                auth_code_holder["error_description"] = params.get(
                    "error_description", [None]
                )[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                if auth_code_holder.get("error"):
                    self.wfile.write(
                        f"<html><body style='font-family:sans-serif;text-align:center;"
                        f"padding:60px'><h1>Login failed</h1>"
                        f"<p>{auth_code_holder.get('error_description', 'Unknown error')}</p>"
                        f"</body></html>".encode()
                    )
                else:
                    self.wfile.write(
                        b"<html><body style='font-family:sans-serif;text-align:center;"
                        b"padding:60px'><h1>Login successful!</h1>"
                        b"<p>You can close this tab and return to the terminal.</p>"
                        b"</body></html>"
                    )

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
        webbrowser.open(auth_url)

        server.handle_request()
        server.server_close()

        if auth_code_holder.get("error"):
            raise ValueError(
                f"OAuth error: {auth_code_holder['error']} - "
                f"{auth_code_holder.get('error_description', 'no details')}"
            )

        if auth_code_holder.get("state") != self.state:
            raise ValueError("OAuth state mismatch - possible CSRF attack")
        if not auth_code_holder.get("code"):
            raise ValueError("No authorization code received")

        self._exchange_code(auth_code_holder["code"])
        result = self._build_result()
        self._save_tokens()
        return result

    def load_or_login(self) -> CodexAuthResult:
        """Load cached tokens and refresh, or start fresh login if needed."""
        if self._load_tokens():
            try:
                self._refresh_tokens()
                self._save_tokens()
                return self._build_result()
            except Exception:
                if TOKEN_CACHE_FILE.exists():
                    try:
                        TOKEN_CACHE_FILE.unlink()
                    except OSError:
                        # Continue to fresh login even if stale cache cleanup fails.
                        pass

        return self.login()

    def logout(self):
        """Clear cached tokens."""
        if TOKEN_CACHE_FILE.exists():
            TOKEN_CACHE_FILE.unlink()
        self.tokens = None
