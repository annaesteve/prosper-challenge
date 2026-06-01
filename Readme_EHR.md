# Prosper Health — AI Voice Appointment Scheduler

A Pipecat voice agent wired to a lightweight in-memory EHR (Electronic Health Record) API.

---

## Architecture

```
Browser / WebRTC client
        │
        ▼
  Pipecat bot.py  ──── OpenAI LLM (with tool calling) ────▶ ehr.py (FastAPI)
        │                                                          │
   ElevenLabs STT/TTS                                    In-memory patient DB
                                                         Slots & appointments
```

The components:

| File | Purpose |
|------|---------|
| `ehr.py` | FastAPI service — the "EHR". Manages patients, slots, appointments. |
| `bot.py` | Pipecat voice agent. Calls the EHR via HTTP tool calls during conversation. |

---

## EHR API Reference

Base URL: `http://localhost:8000`

### Identify a patient
```
GET /patients/search?name=Alice+Johnson
GET /patients/search?dob=1985-03-12
GET /patients/search?name=Alice&dob=1985-03-12
```

### List available slots
```
GET /slots
GET /slots?date=2025-06-02
GET /slots?specialty=Cardiology
GET /slots?date=2025-06-02&specialty=General+Practice
```

### Book an appointment
```
POST /appointments
Content-Type: application/json

{
  "patient_id": "P001",
  "slot_id": "SL-2025-06-02-D001-0900",
  "reason": "Annual check-up"
}
```

### View patient appointments
```
GET /appointments/P001
```

### Cancel an appointment
```
DELETE /appointments/APT-1A2B3C4D
```

---

## Seed data

The EHR starts with four patients:

| ID | Name | DOB | Insurance |
|----|------|-----|-----------|
| P001 | Alice Johnson | 1985-03-12 | BlueCross |
| P002 | Bob Martinez | 1972-07-24 | Aetna |
| P003 | Carol Smith | 1990-11-05 | UnitedHealth |
| P004 | David Lee | 1965-01-30 | Medicare |

And four doctors:

| ID | Name | Specialty |
|----|------|-----------|
| D001 | Dr. Sarah Chen | General Practice |
| D002 | Dr. James Patel | Cardiology |
| D003 | Dr. Emily Torres | Dermatology |
| D004 | Dr. Michael Brown | Orthopedics |

Slots are auto-generated for the next 7 working days at 9 AM, 10 AM, 11 AM, 2 PM, 3 PM, and 4 PM for each doctor.

---

## Setup

### 1. Environment variables

Copy `.env.example` to `.env` and fill in:

```
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
EHR_BASE_URL=http://localhost:8000   # optional, this is the default
```

### 2. Install dependencies

The EHR needs FastAPI, uvicorn, and httpx. Add them to `pyproject.toml`:

```toml
[project]
dependencies = [
    "pipecat-ai[elevenlabs,openai,silero,webrtc]",
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "python-dotenv",
]
```

Then install:

```bash
uv sync
```

### 3. Run

**Terminal 1 — EHR service:**
```bash
uv run python ehr.py
# → Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Voice bot:**
```bash
uv run bot.py
```

Open the WebRTC client URL printed in Terminal 2 and speak to the agent.

---

## Example conversation

> **Bot:** Hi there! Welcome to Prosper Health. I'm your scheduling assistant. Could I get your name and date of birth to look you up in our system?

> **Patient:** Sure, I'm Alice Johnson, born March 12th, 1985.

> **Bot:** *(calls `identify_patient`)* Found you, Alice! How can I help you today — would you like to schedule a new appointment or view an existing one?

> **Patient:** I'd like to see a cardiologist next week.

> **Bot:** *(calls `list_available_slots` with specialty=Cardiology)* Dr. James Patel has availability on Tuesday the 3rd at 9 AM or 2 PM, and on Thursday the 5th at 10 AM. Which works best for you?

> **Patient:** Tuesday at 2 PM, please.

> **Bot:** Just to confirm — I'll book you with Dr. James Patel on Tuesday June 3rd at 2 PM. Shall I go ahead?

> **Patient:** Yes.

> **Bot:** *(calls `book_appointment`)* Done! Your appointment is confirmed — Dr. Patel, Tuesday June 3rd at 2 PM. Is there anything else I can help with?

---

## Extending the EHR

The in-memory store in `ehr.py` is intentionally simple. To make it production-ready:

- Replace the `PATIENTS`, `_SLOTS`, and `_APPOINTMENTS` dicts with a real database (SQLite → PostgreSQL).
- Add authentication (OAuth2 / API keys) to the FastAPI routes.
- Persist slot generation so it survives restarts.
- Add patient registration and insurance verification endpoints.
- Wire real calendar / EMR systems (Epic, Cerner) via FHIR APIs.