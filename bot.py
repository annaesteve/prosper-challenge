#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Voice Agent — Prosper Health Appointment Scheduler.

The bot can:
  1. Identify a patient by name and/or date of birth (via EHR API)
  2. Register a new patient if they are not found
  3. List available appointment slots (filtered by date / specialty)
  4. Book an appointment for the identified patient
  5. Cancel an existing appointment

Required environment variables:
  ELEVENLABS_API_KEY
  OPENAI_API_KEY
  EHR_BASE_URL  (optional, defaults to http://localhost:8000)
"""

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot (Prosper Health Appointment Scheduler)...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("✅ Local Smart Turn Analyzer V3 loaded")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

EHR_BASE_URL = os.environ.get("EHR_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# EHR tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

EHR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "identify_patient",
            "description": (
                "Look up a patient in the EHR. "
                "ALWAYS collect the patient's first AND last name before calling this — never call it with only a first name. "
                "dob is optional but helps disambiguate common names. "
                "If not found, offer to register them as a new patient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name (first and last), e.g. 'Alice Johnson'. REQUIRED.",
                    },
                    "dob": {
                        "type": "string",
                        "description": "Date of birth in YYYY-MM-DD format, e.g. '1985-03-12'. Optional.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_patient",
            "description": (
                "Register a brand-new patient who does not yet exist in the EHR. "
                "Only call this after identify_patient returned no results AND the patient "
                "has explicitly confirmed they want to be registered. "
                "first_name, last_name, and dob are required. "
                "Collect phone, email, and insurance politely before calling — they are optional "
                "but useful. Returns the new patient record with their assigned patient ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "Patient's first name"},
                    "last_name":  {"type": "string", "description": "Patient's last name"},
                    "dob":        {"type": "string", "description": "Date of birth YYYY-MM-DD"},
                    "phone":      {"type": "string", "description": "Phone number (optional)"},
                    "email":      {"type": "string", "description": "Email address (optional)"},
                    "insurance":  {"type": "string", "description": "Insurance provider (optional)"},
                },
                "required": ["first_name", "last_name", "dob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_slots",
            "description": (
                "Return available appointment slots at Prosper Health clinic. "
                "Optionally filter by date (YYYY-MM-DD) and/or medical specialty. "
                "Specialties available: General Practice, Cardiology, Dermatology, Orthopedics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Filter by a specific date YYYY-MM-DD",
                    },
                    "specialty": {
                        "type": "string",
                        "description": "Filter by medical specialty",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an appointment for a patient. "
                "Requires a patient_id (from identify_patient or register_patient) "
                "and a slot_id (from list_available_slots). "
                "Always confirm details with the patient before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's ID, e.g. 'P001'",
                    },
                    "slot_id": {
                        "type": "string",
                        "description": "The slot ID to book",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief description of the reason for the visit",
                    },
                },
                "required": ["patient_id", "slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_patient_appointments",
            "description": (
                "List all existing appointments for a patient. "
                "Use this to show upcoming appointments or find an appointment ID before cancelling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's ID",
                    },
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing appointment by its appointment ID. "
                "First call list_patient_appointments to get the appointment ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "The appointment ID to cancel, e.g. 'APT-1A2B3C4D'",
                    },
                },
                "required": ["appointment_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# EHR tool executor
# ---------------------------------------------------------------------------


async def execute_ehr_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Call the EHR HTTP API and return a JSON string result."""
    async with httpx.AsyncClient(base_url=EHR_BASE_URL, timeout=10.0) as client:
        try:
            if tool_name == "identify_patient":
                params = {k: v for k, v in args.items() if v}
                resp = await client.get("/patients/search", params=params)

            elif tool_name == "register_patient":
                resp = await client.post("/patients", json=args)

            elif tool_name == "list_available_slots":
                params = {k: v for k, v in args.items() if v}
                resp = await client.get("/slots", params=params)

            elif tool_name == "book_appointment":
                resp = await client.post("/appointments", json=args)

            elif tool_name == "list_patient_appointments":
                patient_id = args["patient_id"]
                resp = await client.get(f"/appointments/{patient_id}")

            elif tool_name == "cancel_appointment":
                appt_id = args["appointment_id"]
                resp = await client.delete(f"/appointments/{appt_id}")

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            resp.raise_for_status()
            return json.dumps(resp.json())

        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", str(exc))
            return json.dumps({"error": detail, "status_code": exc.response.status_code})
        except Exception as exc:
            logger.error(f"EHR tool error ({tool_name}): {exc}")
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly and professional AI scheduling assistant for Prosper Health clinic.

Your primary responsibilities:
1. Greet the patient and collect their FULL name (first AND last name) before doing anything else.
2. Look them up in the system. If not found, offer to register them.
3. Help them view, schedule, or cancel appointments.
4. Confirm all details clearly before completing any action.

IDENTIFICATION RULES — follow these strictly:
- NEVER call identify_patient with only a first name. Always wait until you have both first and last name.
- If the patient gives only their first name, ask: "Could you also give me your last name?"
- Once you have the full name, call identify_patient immediately. DOB is optional — only ask for it if the name search returns multiple matches.

NO APPOINTMENTS RULE — very important:
- If list_patient_appointments returns an empty list or a message saying no appointments exist, respond IMMEDIATELY and naturally: "You don't have any appointments booked at the moment. Would you like to schedule one?"
- Do NOT pause, wait, or call any other tool. Speak straight away.

REGISTRATION FLOW (only when patient is not found):
- Tell them they are not in the system and ask if they would like to register.
- If yes, you already have their name — ask for date of birth next.
- Then optionally ask for phone, email, and insurance — one at a time, keeping it brief.
- Confirm all details before calling register_patient.
- After registering, offer to book an appointment straight away.

GENERAL GUIDELINES:
- When listing slots, present only 3-4 options in natural language: "Dr. Chen has availability on Monday at 9 AM or 10 AM."
- Always confirm appointment details before calling book_appointment.
- After any action, summarise what was done in one sentence.
- Be concise — this is a voice conversation. No bullet points, no long lists.
- Never share another patient's information.

Available specialties: General Practice, Cardiology, Dermatology, Orthopedics.
Clinic hours: Monday–Friday, 9 AM – 5 PM.
"""


# ---------------------------------------------------------------------------
# Bot entry point
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Prosper Health appointment bot")

    elevenlabs_key = os.environ["ELEVENLABS_API_KEY"]
    stt = ElevenLabsRealtimeSTTService(api_key=elevenlabs_key)
    tts = ElevenLabsTTSService(
        api_key=elevenlabs_key,
        voice_id="SAz9YHcvj6GT2YYXdXww",
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        tools=EHR_TOOLS,
    )

    @llm.event_handler("on_tool_call")
    async def on_tool_call(llm_service, tool_call):
        tool_name = tool_call.function_name
        try:
            args = json.loads(tool_call.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Tool call: {tool_name}({args})")
        result = await execute_ehr_tool(tool_name, args)
        logger.info(f"Tool result: {result[:200]}...")

        await llm_service.push_tool_result(tool_call, result)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )

    rtvi = RTVIProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append(
            {
                "role": "system",
                "content": (
                    "The patient has just connected. Greet them warmly and introduce yourself "
                    "as the scheduling assistant for Prosper Health clinic. "
                    "Ask for their name and date of birth so you can look up their record."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()