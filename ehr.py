"""
ehr.py — Simple Electronic Health Record (EHR) HTTP API
========================================================
Runs on http://localhost:8000
Database: SQLite (ehr.db, created automatically on first run)
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DB_PATH = "ehr.db"

app = FastAPI(title="Prosper Health EHR", version="1.0.0")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row) -> dict:
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Schema + seed data (runs once on startup)
# ---------------------------------------------------------------------------

DOCTORS = [
    ("D001", "Dr. Sarah Chen",    "General Practice"),
    ("D002", "Dr. James Patel",   "Cardiology"),
    ("D003", "Dr. Emily Torres",  "Dermatology"),
    ("D004", "Dr. Michael Brown", "Orthopedics"),
]

SEED_PATIENTS = [
    ("P001", "Alice",  "Johnson",  "1985-03-12", "555-100-0001", "alice.johnson@email.com",  "BlueCross"),
    ("P002", "Bob",    "Martinez", "1972-07-24", "555-100-0002", "bob.martinez@email.com",   "Aetna"),
    ("P003", "Carol",  "Smith",    "1990-11-05", "555-100-0003", "carol.smith@email.com",    "UnitedHealth"),
    ("P004", "David",  "Lee",      "1965-01-30", "555-100-0004", "david.lee@email.com",      "Medicare"),
]

SLOT_TIMES = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id          TEXT PRIMARY KEY,
                first_name  TEXT NOT NULL,
                last_name   TEXT NOT NULL,
                dob         TEXT NOT NULL,
                phone       TEXT,
                email       TEXT,
                insurance   TEXT
            );

            CREATE TABLE IF NOT EXISTS doctors (
                id        TEXT PRIMARY KEY,
                name      TEXT NOT NULL,
                specialty TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS slots (
                id          TEXT PRIMARY KEY,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                doctor_id   TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                specialty   TEXT NOT NULL,
                booked      INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id           TEXT PRIMARY KEY,
                patient_id   TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                slot_id      TEXT NOT NULL,
                date         TEXT NOT NULL,
                time         TEXT NOT NULL,
                doctor_name  TEXT NOT NULL,
                specialty    TEXT NOT NULL,
                reason       TEXT,
                status       TEXT NOT NULL DEFAULT 'confirmed',
                created_at   TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (slot_id)    REFERENCES slots(id)
            );
        """)

        # Seed doctors
        conn.executemany(
            "INSERT OR IGNORE INTO doctors VALUES (?,?,?)", DOCTORS
        )

        # Seed patients
        conn.executemany(
            "INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?,?)", SEED_PATIENTS
        )

        # Seed slots for the next 7 working days (skip if already present)
        today = date.today()
        for offset in range(1, 8):
            day = today + timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            for doc_id, doc_name, specialty in DOCTORS:
                for t in SLOT_TIMES:
                    slot_id = f"SL-{day.isoformat()}-{doc_id}-{t.replace(':','')}"
                    conn.execute(
                        "INSERT OR IGNORE INTO slots VALUES (?,?,?,?,?,?,0)",
                        (slot_id, day.isoformat(), t, doc_id, doc_name, specialty),
                    )


init_db()

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class PatientRegistration(BaseModel):
    first_name: str
    last_name: str
    dob: str                          # YYYY-MM-DD
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance: Optional[str] = None


class AppointmentRequest(BaseModel):
    patient_id: str
    slot_id: str
    reason: Optional[str] = "General consultation"

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/patients/search")
def search_patients(
    name: Optional[str] = Query(None),
    dob:  Optional[str] = Query(None),
):
    if not name and not dob:
        raise HTTPException(400, "Provide at least 'name' or 'dob'.")
    with get_db() as conn:
        if name and dob:
            rows = conn.execute(
                "SELECT * FROM patients WHERE (first_name || ' ' || last_name) LIKE ? AND dob = ?",
                (f"%{name}%", dob),
            ).fetchall()
        elif name:
            rows = conn.execute(
                "SELECT * FROM patients WHERE (first_name || ' ' || last_name) LIKE ?",
                (f"%{name}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM patients WHERE dob = ?", (dob,)
            ).fetchall()
    patients = [dict(r) for r in rows]
    return {"found": bool(patients), "patients": patients}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Patient not found.")
    return dict(row)


@app.post("/patients", status_code=201)
def register_patient(req: PatientRegistration):
    """Register a brand-new patient and return their generated ID."""
    with get_db() as conn:
        # Duplicate check: same full name + DOB
        existing = conn.execute(
            "SELECT id FROM patients WHERE first_name = ? AND last_name = ? AND dob = ?",
            (req.first_name, req.last_name, req.dob),
        ).fetchone()
        if existing:
            raise HTTPException(409, f"Patient already exists with ID {existing['id']}.")

        # Generate a numeric ID that continues from the highest existing one
        row = conn.execute("SELECT MAX(CAST(SUBSTR(id,2) AS INTEGER)) FROM patients").fetchone()
        next_num = (row[0] or 0) + 1
        patient_id = f"P{next_num:03d}"

        conn.execute(
            "INSERT INTO patients VALUES (?,?,?,?,?,?,?)",
            (patient_id, req.first_name, req.last_name, req.dob,
             req.phone, req.email, req.insurance),
        )

    return {
        "registered": True,
        "id": patient_id,
        "first_name": req.first_name,
        "last_name": req.last_name,
        "dob": req.dob,
        "phone": req.phone,
        "email": req.email,
        "insurance": req.insurance,
    }



def list_slots(
    date:      Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
):
    query = "SELECT * FROM slots WHERE booked = 0"
    params: list = []
    if date:
        query += " AND date = ?"
        params.append(date)
    if specialty:
        query += " AND specialty LIKE ?"
        params.append(f"%{specialty}%")
    query += " ORDER BY date, time"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"slots": [dict(r) for r in rows]}


@app.post("/appointments", status_code=201)
def book_appointment(req: AppointmentRequest):
    with get_db() as conn:
        patient = conn.execute("SELECT * FROM patients WHERE id = ?", (req.patient_id,)).fetchone()
        if not patient:
            raise HTTPException(404, "Patient not found.")
        slot = conn.execute("SELECT * FROM slots WHERE id = ?", (req.slot_id,)).fetchone()
        if not slot:
            raise HTTPException(404, "Slot not found.")
        if slot["booked"]:
            raise HTTPException(409, "Slot is already booked.")

        appt_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
        patient_name = f"{patient['first_name']} {patient['last_name']}"
        now = datetime.utcnow().isoformat()

        conn.execute(
            """INSERT INTO appointments
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (appt_id, req.patient_id, patient_name, req.slot_id,
             slot["date"], slot["time"], slot["doctor_name"],
             slot["specialty"], req.reason, "confirmed", now),
        )
        conn.execute("UPDATE slots SET booked = 1 WHERE id = ?", (req.slot_id,))

    return {
        "id": appt_id, "patient_id": req.patient_id, "patient_name": patient_name,
        "slot_id": req.slot_id, "date": slot["date"], "time": slot["time"],
        "doctor_name": slot["doctor_name"], "specialty": slot["specialty"],
        "reason": req.reason, "status": "confirmed", "created_at": now,
    }


@app.get("/appointments/{patient_id}")
def list_appointments(patient_id: str):
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone():
            raise HTTPException(404, "Patient not found.")
        rows = conn.execute(
            "SELECT * FROM appointments WHERE patient_id = ? ORDER BY date, time",
            (patient_id,),
        ).fetchall()
    appts = [dict(r) for r in rows]
    if not appts:
        return {"appointments": [], "message": "This patient has no appointments on record."}
    return {"appointments": appts}


@app.delete("/appointments/{appointment_id}")
def cancel_appointment(appointment_id: str):
    with get_db() as conn:
        appt = conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
        if not appt:
            raise HTTPException(404, "Appointment not found.")
        if appt["status"] == "cancelled":
            raise HTTPException(409, "Appointment already cancelled.")
        conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,)
        )
        conn.execute("UPDATE slots SET booked = 0 WHERE id = ?", (appt["slot_id"],))
    return {"cancelled": True, "appointment": dict(appt)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")