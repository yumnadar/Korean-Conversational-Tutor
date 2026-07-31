"""
Minimal web UI for the Korean Conversation Tutor.

"""
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
app = FastAPI(title="Korean Conversation Tutor")
HERE = Path(__file__).resolve().parent
SESSIONS: dict[str, dict] = {}

@app.get("/")
def index():
    """Serve the single-page frontend."""
    return FileResponse(HERE / "index.html")

@app.get("/api/registers")
def list_registers():
    """The six speech levels, for the opening menu."""
    return [
        {
            "key": key,
            "label": reg["label"],
            "hint": reg.get("menu_hint", ""),
            "note": reg.get("note", ""),
        }
        for key, reg in REGISTERS.items()
    ]

class ScenarioRequest(BaseModel):
    register: str

@app.post("/api/scenarios")
def make_scenarios(req: ScenarioRequest):
    """Ask the Brain for four roleplay scenarios suited to the chosen register."""
    if req.register not in REGISTERS:
        raise HTTPException(400, f"Unknown register: {req.register}")
    return {"scenarios": generate_scenarios(req.register)}

class StartRequest(BaseModel):
    register: str
    scenario: dict

@app.post("/api/start")
def start(req: StartRequest):
    """Open a scene: create a session and have the character speak first."""
    if req.register not in REGISTERS:
        raise HTTPException(400, f"Unknown register: {req.register}")
    history: list = []
    plan = plan_turn(req.register, req.scenario, history, None)
    opening = (plan["next_line"].get("draft") or "").strip()
    history.append(("Character", opening))
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {
        "register": req.register,
        "scenario": req.scenario,
        "history": history,
        "progress": Progress(),
    }
    return {"session_id": sid, "opening": opening}

class MessageRequest(BaseModel):
    session_id: str
    text: str

@app.post("/api/message")
def message(req: MessageRequest):
    """One conversation turn: evaluate the student's line, reply in character."""
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Start a new scenario.")
    student = req.text.strip()
    if not student:
        raise HTTPException(400, "Message is empty.")
    session["history"].append(("Student", student))
    plan = plan_turn(
        session["register"], session["scenario"],
        session["history"], student, session["progress"],
    )
    reply = (plan["next_line"].get("draft") or "").strip() or "죄송합니다, 다시 말씀해 주시겠어요?"
    session["history"].append(("Character", reply))
    session["progress"].record(plan.get("feedback"))
    return {"reply": reply, "feedback": plan.get("feedback")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
