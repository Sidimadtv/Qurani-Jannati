"""Qurani Jannati - self-hosted recitation correction API.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

To enable the AI speech engine (open-source faster-whisper):
    pip install -r requirements-whisper.txt

Endpoints
---------
GET  /health                 -> engine status
POST /api/v1/correct         -> multipart: audio + expected_text + ayah/surah
POST /api/v1/demo            -> run the aligner on sample text (no audio)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import correction
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

API_KEY = os.environ.get("QURANI_JANNATI_API_KEY", "")
ALLOWED_ORIGINS = os.environ.get(
    "QURANI_JANNATI_CORS_ORIGINS", "*"
).split(",")

app = FastAPI(
    title="Qurani Jannati Recitation Correction API",
    description="Self-hosted AI recitation correction for the Qurani Jannati app.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(authorization: Optional[str]) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Set QURANI_JANNATI_API_KEY on the server and pass it in the app settings.",
        )


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version, **correction.engine_status()}


@app.post("/api/v1/correct")
async def correct(
    session_id: str = Form(...),
    ayah: int = Form(...),
    surah: int = Form(0),
    expected_text: str = Form(...),
    language: str = Form("ar"),
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    _check_auth(authorization)

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(
        prefix="qj-", suffix=suffix, delete=False
    )
    try:
        tmp.write(await audio.read())
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()

    try:
        result = correction.correct_audio(
            tmp.name,
            expected_text,
            session_id=session_id,
            ayah=ayah,
            surah=surah,
            language=language,
        )
        status = 200 if result.get("engine") == "unavailable" else 200
        return JSONResponse(status_code=status, content=result)
    finally:
        _cleanup(tmp.name)


@app.post("/api/v1/demo")
async def demo(
    expected_text: str = Form(
        "قُلْ هُوَ اللَّهُ أَحَدٌ ۝ اللَّهُ الصَّمَدُ"
    ),
    received_text: str = Form("قُل هو الله أحد الله الصَمَد"),
    authorization: Optional[str] = Header(None),
) -> JSONResponse:
    """Run the aligner on sample transcripts so you can test the engine
    without recording audio."""
    _check_auth(authorization)
    result = correction.compute_mistakes(
        expected_text,
        received_text,
        session_id="demo",
        ayah=1,
        surah=112,
        audio_duration_ms=4000,
    )
    result["engine"] = "aligner"
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
