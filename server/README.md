# Qurani Jannati — Self-hosted Recitation Correction API

The backend for the **Recitation** feature in the Qurani Jannati app (Tarteel-style,
but self-hosted — no third-party SaaS, no Tarteel servers). It transcribes your
recitation and reports mistakes.

## How correction works

1. The app records your voice and uploads it together with the ayah you recited.
2. This server transcribes the audio (open-source `faster-whisper`, Arabic pass).
3. A word-level aligner in `correction.py` compares the recognized transcript to
   the expected ayah text and emits mistakes:
   - `INCORRECT_TASHKEEL` — letters match, harakat differ
   - `INCORRECT_WORDS` — a word was said incorrectly
   - `MISSED_WORDS` — a word was skipped
   - `EXTRA_WORDS` — extra words were spoken
   - `PEEKED_WORDS` — reserved (mirrors the Tarteel enum)
4. Mistakes come back with word positions + timestamps so the app can show you
   exactly what to practice.

## Endpoints

| Method | Path            | Purpose                                        |
|--------|-----------------|------------------------------------------------|
| GET    | `/health`       | Engine status (`available` / `unavailable`)     |
| POST   | `/api/v1/correct` | Multipart audio + `expected_text`, `ayah`, `surah` |
| POST   | `/api/v1/demo`  | Test the aligner on sample text (no audio)      |

## Run locally (no AI)

```bash
cd server
python -m venv venv
venv\Scripts\activate        # Windows:  .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

`http://localhost:8000/health` should return `{"status":"ok", ..., "available": false}`.
The API still records + saves sessions; it just can't score accuracy yet.

## Enable the AI speech engine

```bash
pip install -r requirements-whisper.txt   # installs faster-whisper + torch (CPU)
```

Then restart. `/health` will report `"available": true` and `/api/v1/correct`
will transcribe and score your recitation. Models are downloaded automatically
on first use.

## Secure it (recommended for any public deploy)

Create `.env` (or export these) and run with `--env-file`:

```bash
QURANI_JANNATI_API_KEY=your-secret-key
QURANI_JANNATI_CORS_ORIGINS=*
PORT=8000
```

The app must send `Authorization: Bearer <your-secret-key>`.

## Deploy

### A) Free/cheap host (no AI engine — records + saves only)
- **Render** — New *Web Service* → root `server/`, build `pip install -r requirements.txt`,
  start `uvicorn main:app --host 0.0.0.0 --port $PORT`. Add the env vars above.
- **Railway** — same settings. `PORT` is set automatically.

### B) VPS (full AI — recommended)
```bash
git clone https://github.com/Sidimadtv/Qurani-Jannati.git
cd Qurani-Jannati/server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-whisper.txt
# systemd unit:
ExecStart=/opt/qj/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

### C) Docker
```bash
docker build -t qj-server server
docker run -d -p 8000:8000 -e QURANI_JANNATI_API_KEY=your-key qj-server
```

## Point the app at your server

1. Open the app → **Recitation** dashboard (home → Recitation tile).
2. Tap the **settings (gear)** icon in the top-right.
3. Set **Server URL** to `https://your-host` (or `http://<PC-LAN-IP>:8000`
   when testing from your phone on the same Wi-Fi).
4. Set the **API key** you configured above (leave blank if you didn't set one).

Your recordings sync to your own server. The dashboard shows streaks, daily
goal, accuracy and every saved session with its mistakes.
