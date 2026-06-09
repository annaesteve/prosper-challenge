# SOLUTION.md — Prosper challenge

## Framework

This is a voice AI agent that lets patients call into Prosper Health clinic and:
- identify themselves by name and optionally date of birth
- register as a new patient if they are not in the system
- browse available appointment slots by specialty and date
- book a confirmed appointment
- view or cancel existing appointments

The system has two processes:

```
1. Browser (WebRTC)
bot.py  (Pipecat pipeline)
  1.1 ElevenLabs STT  — speech → text
  1.2 OpenAI GPT-4o   — conversation + tool calling
  1.3 ElevenLabs TTS  — text → speech

2. HTTP (localhost:8000)
ehr.py  (FastAPI + SQLite)
    /patients/search   — identify patient
    /patients          — register patient
    /slots             — available appointment slots
    /appointments      — book / list / cancel
    ehr.db             — persistent SQLite database
```

---

## Key decisions and trade-offs

### 1. Separating the EHR from the bot

The EHR runs as a standalone FastAPI service rather than being imported directly into `bot.py`. This means any future frontend, admin dashboard, or third-party system can also call the same API without touching the bot code. Additionally the EHR can be swapped or modified without need of touching bot.py and just by changing a single `EHR_BASE_URL` environment variable.

The trade-off is one extra network hop per tool call (localhost HTTP). In practice this adds under 1ms and is completely invisible in a voice conversation.

### 2. SQLite over in-memory dictionaries

At first I started using dicts for comfort. Those vanish on restart, every booking would be lost if the server crashed or was redeployed. SQLite gives full persistence with zero infrastructure: no separate database server, no connection pooling config, no Docker service. It is a single file (`ehr.db`) that survives restarts and can be copied as a backup.

WAL (Write-Ahead Logging) mode is enabled so concurrent reads don't block writes, which matters when multiple HTTP requests arrive at the same time.

### 3. Tool calling for EHR actions

Rather than hardcoding booking logic inside the system prompt, each action (identify, register, list slots, book, cancel) is an OpenAI function-calling tool with a typed schema. This has several benefits:

- The LLM decides *when* to call a tool based on conversational context.
- The schema enforces types before the EHR even sees the request.
- Adding a new capability (e.g. reschedule) is an isolated change: add one tool definition, one EHR endpoint, one executor branch.

The trade-off is that tool calling adds a round-trip to OpenAI's API (the model first returns a tool call, then we call the EHR, then we send the result back). This adds roughly 300–600ms per action. Acceptable for a voice appointment scheduler.

### 4. Mandatory full name before identification

Early testing showed the bot would sometimes call `identify_patient` with only a first name, producing spurious matches. This was fixed at two levels simultaneously:
- The tool schema marks `name` as required, so OpenAI's function calling layer enforces it.
- The system prompt explicitly instructs the bot to ask for the last name before calling anything.

Dual enforcement is intentional: LLM instructions can be forgotten deep in a long conversation; schema constraints cannot.

### 5. Explicit "no appointments" message from the EHR

When a patient has no appointments, the EHR returns `{"appointments": [], "message": "This patient has no appointments on record."}` instead of just an empty array. An empty array caused the LLM to stall, it had nothing concrete to say and would sometimes loop or remain silent. A pre-composed message gives it an immediate, speakable string.

### 6. Slot generation strategy

Slots are generated for the next 7 working days when the EHR starts, using `INSERT OR IGNORE` so restarts never duplicate them. The simplest possible approach for a demo. The obvious gap is that as days pass, old slots expire and new ones are never added unless the server restarts.

---

## Potential improvements

### Latency

**Problem**: A single patient action (e.g. booking) involves: STT → LLM (tool decision) → EHR HTTP call → LLM (response generation) → TTS. End-to-end this is typically 2–4 seconds, which is noticeable in a voice conversation.

**Options**:

- **Streaming TTS**: Start speaking the first sentence while the rest is still being generated. 
- **EHR response caching**: Cache slot availability per specialty per day with a short TTL (e.g. 30 seconds). Slot lists rarely change mid-call. This eliminates one HTTP round-trip for the most common query.
- **Smaller STT model**: ElevenLabs Realtime STT is high quality but not the fastest. For a scheduling bot that only needs to recognise names, dates, and simple phrases, a smaller Whisper `base` or `small` model running locally would be faster with acceptable accuracy.
- **Parallel tool pre-fetching**: When the patient is identified, immediately (and silently) fetch their appointments in the background. By the time they ask "what do I have booked?", the answer is already there.

### Reliability

**Problem**: The bot depends on ElevenLabs (STT + TTS) and OpenAI (LLM). If either goes down, the agent is completely silent.

**Options**:

- **Fallback LLM**: Configure a secondary provider (e.g. Anthropic Claude, or a locally-hosted Ollama model) and switch automatically if the primary returns a 5xx or times out. The tool schema is OpenAI-compatible with Anthropic's API.
- **Fallback TTS**: If ElevenLabs TTS fails, fall back to a local Kokoro or Coqui TTS instance. Voice quality degrades but the call continues.
- **Fallback STT**: If ElevenLabs STT fails, fall back to local Whisper. Latency increases but does not drop to zero.
- **EHR health check on startup**: `bot.py` should ping `GET /health` on the EHR before accepting calls and fail fast with a clear log message rather than silently producing broken conversations.

### Evaluation

**Problem**: The agent is a voice conversation. You cannot unit-test "did it sound natural?" and you cannot catch hallucinations by reading source code. The only way to know the agent is behaving correctly is to simulate real calls.

**Approach 1 — Scripted conversation replay**

Write a test harness that injects text directly into the LLM context (bypassing STT/TTS) and asserts on the tool calls the model makes and the responses it produces:

```python
async def test_new_patient_registration():
    # Feed messages as if a patient typed them
    result = await run_conversation([
        "Hi, I'm Jane Doe",          # only first + last name, no DOB
        "Sure, I'd like to register",
        "My DOB is 1990-05-20",
        "No phone, no email, Aetna insurance",
        "Yes, confirm",
    ])
    assert result.tool_calls_made == [
        "identify_patient",    # called after full name given
        "register_patient",    # called after confirmation
    ]
    assert result.tool_calls_made[0].args["name"] == "Jane Doe"
    assert "registered" in result.final_utterance.lower()
```

This can run in CI on every commit in under 10 seconds with no real API calls (mock the EHR and LLM).

**Approach 2 — LLM-as-judge evaluation**

Run the agent against a suite of synthetic patient scenarios and have a second LLM grade the conversation:

```python
SCENARIOS = [
    {"patient": "Alice Johnson, DOB 1985-03-12", "intent": "book cardiology next week"},
    {"patient": "Unknown Person",                "intent": "book appointment → should register"},
    {"patient": "Bob Martinez",                  "intent": "cancel his only appointment"},
    {"patient": "Carol Smith",                   "intent": "check appointments → has none"},
]

# For each scenario, simulate the full conversation, then ask GPT-4o:
# "Did the agent: (1) identify the patient correctly? (2) complete the requested action?
#  (3) hallucinate any information? (4) respond within reasonable turn length?"
```

This catches regressions after system prompt changes and after upgrading the underlying model.

**Approach 3 — Structured output assertions**

Force the agent to produce a structured JSON summary at the end of every call:

```json
{
  "patient_identified": true,
  "patient_id": "P001",
  "actions_taken": ["list_slots", "book_appointment"],
  "appointment_id": "APT-3F2A1B4C",
  "outcome": "booked"
}
```

This summary can be logged and compared against the EHR state to detect discrepancies (e.g. the agent said it booked something but the EHR shows nothing).

