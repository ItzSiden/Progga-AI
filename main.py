"""
EduLumina Backend — FastAPI
Handles: Auth, RAG, Knowledge Base CRUD, Admin Panel, API routing
"""

import os
import re
import time
import json
import httpx
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

import db

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(title="EduLumina API", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

# ── Config from env ───────────────────────────────────────────

JWT_SECRET      = os.environ.get("JWT_SECRET", "change-this-in-production")
ADMIN_EMAIL     = os.environ.get("ADMIN_EMAIL", "admin@edulumina.com")
ADMIN_PASSWORD  = os.environ.get("ADMIN_PASSWORD", "admin123")  # hash this!
JWT_EXPIRE_DAYS = int(os.environ.get("JWT_EXPIRE_DAYS", "30"))

# Free API keys (from env, comma-separated if multiple)
GEMINI_KEYS   = [k.strip() for k in os.environ.get("GEMINI_KEYS", "").split(",") if k.strip()]
GROQ_KEYS     = [k.strip() for k in os.environ.get("GROQ_KEYS", "").split(",") if k.strip()]
CEREBRAS_KEYS = [k.strip() for k in os.environ.get("CEREBRAS_KEYS", "").split(",") if k.strip()]

# ── Rate limiting (in-memory) ─────────────────────────────────

# {ip: deque of timestamps}
_rate_store: dict[str, deque] = defaultdict(lambda: deque())
FREE_LIMIT_PER_DAY  = int(os.environ.get("FREE_LIMIT_PER_DAY", "10"))
PAID_LIMIT_PER_DAY  = int(os.environ.get("PAID_LIMIT_PER_DAY", "100"))

def _check_rate(user_id: str, is_paid: bool) -> tuple[bool, int]:
    """Returns (allowed, remaining)."""
    now   = time.time()
    day   = 86400
    limit = PAID_LIMIT_PER_DAY if is_paid else FREE_LIMIT_PER_DAY
    dq    = _rate_store[user_id]
    # Remove old entries
    while dq and now - dq[0] > day:
        dq.popleft()
    remaining = limit - len(dq)
    if remaining <= 0:
        return False, 0
    dq.append(now)
    return True, remaining - 1

# ── API usage tracking (in-memory, reset on restart) ──────────

_api_stats = {
    "gemini":   {"calls": 0, "errors": 0},
    "groq":     {"calls": 0, "errors": 0},
    "cerebras": {"calls": 0, "errors": 0},
    "total_queries": 0,
    "unique_users": set(),
}

# ── Simple user store (use Supabase in production) ────────────
# {email: {password_hash, is_paid, created_at}}
_users: dict[str, dict] = {}

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _create_token(email: str, is_admin: bool = False) -> str:
    payload = {
        "sub":      email,
        "is_admin": is_admin,
        "exp":      datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

security = HTTPBearer(auto_error=False)

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _decode_token(creds.credentials)

def get_admin(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    user = get_current_user(creds)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user

# ── LLM API router ────────────────────────────────────────────

_key_index = defaultdict(int)

def _next_key(provider: str, keys: list[str]) -> Optional[str]:
    if not keys:
        return None
    idx = _key_index[provider] % len(keys)
    _key_index[provider] += 1
    return keys[idx]

async def _call_groq(prompt: str, context: str) -> str:
    key = _next_key("groq", GROQ_KEYS)
    if not key:
        raise ValueError("No Groq key")
    messages = [
        {"role": "system", "content": _system_prompt(context)},
        {"role": "user",   "content": prompt},
    ]
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 800},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def _call_cerebras(prompt: str, context: str) -> str:
    key = _next_key("cerebras", CEREBRAS_KEYS)
    if not key:
        raise ValueError("No Cerebras key")
    messages = [
        {"role": "system", "content": _system_prompt(context)},
        {"role": "user",   "content": prompt},
    ]
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "llama3.1-8b", "messages": messages, "max_tokens": 800},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def _call_gemini(prompt: str, context: str) -> str:
    key = _next_key("gemini", GEMINI_KEYS)
    if not key:
        raise ValueError("No Gemini key")
    full_prompt = f"{_system_prompt(context)}\n\nQuestion: {prompt}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={key}",
            json={"contents": [{"parts": [{"text": full_prompt}]}]},
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def _system_prompt(context: str) -> str:
    return f"""You are EduLumina, an AI study assistant for Bangladeshi students (Class 6-10, NCTB curriculum).
Answer questions based ONLY on the provided textbook content.
If the answer is not in the context, say so clearly.
Always cite the source (book, page number) in your answer.
Be concise and student-friendly.

TEXTBOOK CONTEXT:
{context}"""

async def ask_llm(prompt: str, context: str) -> tuple[str, str]:
    """
    Try providers in order: Groq → Cerebras → Gemini
    Returns (answer, provider_used)
    """
    providers = [
        ("groq",     _call_groq),
        ("cerebras", _call_cerebras),
        ("gemini",   _call_gemini),
    ]
    last_error = None
    for name, fn in providers:
        if not (GROQ_KEYS if name == "groq" else CEREBRAS_KEYS if name == "cerebras" else GEMINI_KEYS):
            continue
        try:
            _api_stats[name]["calls"] += 1
            answer = await fn(prompt, context)
            return answer, name
        except Exception as e:
            _api_stats[name]["errors"] += 1
            last_error = str(e)
            continue
    raise HTTPException(status_code=503, detail=f"All AI providers failed: {last_error}")


# ════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════

class AuthRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/register")
async def register(body: AuthRequest):
    if body.email in _users:
        raise HTTPException(400, "Email already registered")
    _users[body.email] = {
        "password_hash": _hash(body.password),
        "is_paid":       False,
        "created_at":    datetime.utcnow().isoformat(),
    }
    token = _create_token(body.email)
    return {"token": token, "email": body.email, "is_paid": False}

@app.post("/api/auth/login")
async def login(body: AuthRequest):
    # Admin check
    if body.email == ADMIN_EMAIL and body.password == ADMIN_PASSWORD:
        token = _create_token(body.email, is_admin=True)
        return {"token": token, "email": body.email, "is_admin": True}
    # Regular user
    user = _users.get(body.email)
    if not user or user["password_hash"] != _hash(body.password):
        raise HTTPException(401, "Invalid credentials")
    token = _create_token(body.email)
    return {"token": token, "email": body.email, "is_paid": user["is_paid"]}

@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    email = user["sub"]
    if user.get("is_admin"):
        return {"email": email, "is_admin": True, "is_paid": True}
    u = _users.get(email, {})
    return {"email": email, "is_paid": u.get("is_paid", False)}


# ════════════════════════════════════════════════════════════════
#  CHAT / RAG ROUTE
# ════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str
    version:  Optional[str] = None
    class_num: Optional[int] = None
    subject:  Optional[str] = None
    limit:    int = 5

@app.post("/api/chat")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    email   = user["sub"]
    is_paid = user.get("is_admin") or _users.get(email, {}).get("is_paid", False)

    # Rate limit
    allowed, remaining = _check_rate(email, is_paid)
    if not allowed:
        raise HTTPException(429, f"Daily limit reached. {'Upgrade to Pro for more queries.' if not is_paid else 'Contact support.'}")

    # RAG search
    results = db.rag_search(
        query     = body.question,
        limit     = body.limit,
        version   = body.version,
        class_num = body.class_num,
        subject   = body.subject,
    )
    if not results:
        return {
            "answer":    "Sorry, I couldn't find relevant information in the textbooks for your question.",
            "sources":   [],
            "provider":  "none",
            "remaining": remaining,
        }

    context = db.rag_context(results)
    answer, provider = await ask_llm(body.question, context)

    # Track stats
    _api_stats["total_queries"] += 1
    _api_stats["unique_users"].add(email)

    return {
        "answer":    answer,
        "sources":   [{"source": r["source"], "page": r["page"], "snippet": r["snippet"]} for r in results],
        "provider":  provider,
        "remaining": remaining,
    }


# ════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE ROUTES (Admin only)
# ════════════════════════════════════════════════════════════════

@app.get("/api/kb/list")
async def kb_list(admin: dict = Depends(get_admin)):
    return db.list_all()

@app.get("/api/kb/stats")
async def kb_stats(admin: dict = Depends(get_admin)):
    return db.get_stats()

class SubjectCreate(BaseModel):
    version:   str
    class_num: int
    subject:   str

@app.post("/api/kb/subject")
async def kb_create_subject(body: SubjectCreate, admin: dict = Depends(get_admin)):
    ok, err = db.create_subject(body.version, body.class_num, body.subject)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True, "source": db.source_label(body.version, body.class_num, body.subject)}

@app.delete("/api/kb/subject")
async def kb_delete_subject(body: SubjectCreate, admin: dict = Depends(get_admin)):
    ok, err = db.delete_subject(body.version, body.class_num, body.subject)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True}

@app.get("/api/kb/pages")
async def kb_get_pages(version: str, class_num: int, subject: str, admin: dict = Depends(get_admin)):
    pages = db.get_pages(version, class_num, subject)
    return {"pages": pages, "count": len(pages)}

class BulkImport(BaseModel):
    version:   str
    class_num: int
    subject:   str
    raw_text:  str  # [PAGE_1] ... [PAGE_2] ... format

@app.post("/api/kb/bulk-import")
async def kb_bulk_import(body: BulkImport, admin: dict = Depends(get_admin)):
    count, err = db.bulk_import(body.version, body.class_num, body.subject, body.raw_text)
    if err and count == 0:
        raise HTTPException(400, err)
    return {"imported": count, "warning": err}

class PageAdd(BaseModel):
    version:   str
    class_num: int
    subject:   str
    page:      int
    content:   str

@app.post("/api/kb/page")
async def kb_add_page(body: PageAdd, admin: dict = Depends(get_admin)):
    ok, err = db.add_page(body.version, body.class_num, body.subject, body.page, body.content)
    if not ok:
        raise HTTPException(400, err)
    return {"ok": True}

class PagesSave(BaseModel):
    version:   str
    class_num: int
    subject:   str
    pages:     list[dict]

@app.put("/api/kb/pages")
async def kb_save_pages(body: PagesSave, admin: dict = Depends(get_admin)):
    count, err = db.save_pages(body.version, body.class_num, body.subject, body.pages)
    if err:
        raise HTTPException(500, err)
    return {"saved": count}


# ════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ════════════════════════════════════════════════════════════════

@app.get("/api/admin/stats")
async def admin_stats(admin: dict = Depends(get_admin)):
    return {
        "api": {
            "groq":     _api_stats["groq"],
            "cerebras": _api_stats["cerebras"],
            "gemini":   _api_stats["gemini"],
        },
        "total_queries":  _api_stats["total_queries"],
        "unique_users":   len(_api_stats["unique_users"]),
        "total_users":    len(_users),
        "paid_users":     sum(1 for u in _users.values() if u.get("is_paid")),
        "kb":             db.get_stats(),
        "rate_limits": {
            "free_per_day": FREE_LIMIT_PER_DAY,
            "paid_per_day": PAID_LIMIT_PER_DAY,
        }
    }

@app.get("/api/admin/users")
async def admin_users(admin: dict = Depends(get_admin)):
    return [
        {
            "email":      email,
            "is_paid":    u["is_paid"],
            "created_at": u["created_at"],
        }
        for email, u in _users.items()
    ]

class UserUpdate(BaseModel):
    email:   str
    is_paid: bool

@app.put("/api/admin/user")
async def admin_update_user(body: UserUpdate, admin: dict = Depends(get_admin)):
    if body.email not in _users:
        raise HTTPException(404, "User not found")
    _users[body.email]["is_paid"] = body.is_paid
    return {"ok": True}

@app.delete("/api/admin/user/{email}")
async def admin_delete_user(email: str, admin: dict = Depends(get_admin)):
    if email not in _users:
        raise HTTPException(404, "User not found")
    del _users[email]
    return {"ok": True}

# ── API keys status ───────────────────────────────────────────

@app.get("/api/admin/keys")
async def admin_keys(admin: dict = Depends(get_admin)):
    return {
        "groq":     len(GROQ_KEYS),
        "cerebras": len(CEREBRAS_KEYS),
        "gemini":   len(GEMINI_KEYS),
    }


# ════════════════════════════════════════════════════════════════
#  ADMIN PANEL (HTML)
# ════════════════════════════════════════════════════════════════

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    panel_path = os.path.join(os.path.dirname(__file__), "admin", "index.html")
    if os.path.exists(panel_path):
        with open(panel_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>Admin panel not found</h1>"


# ════════════════════════════════════════════════════════════════
#  HEALTH
# ════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    stats = db.get_stats()
    return {
        "status":   "ok",
        "subjects": stats["total_subjects"],
        "pages":    stats["total_pages"],
        "providers": {
            "groq":     bool(GROQ_KEYS),
            "cerebras": bool(CEREBRAS_KEYS),
            "gemini":   bool(GEMINI_KEYS),
        }
    }
