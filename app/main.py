
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from fastapi import FastAPI, HTTPException, Request
from typing import Optional, Dict, Any
from app.interview_agent import InterviewAgent
from app.services.audio_utils import transcribe_audio, synthesize_text
from app.utils import clean_text, normalize_unicode
import re

app = FastAPI()
agent = InterviewAgent()

# --- ENDPOINTS DE AUDIO (STT/TTS) ---
@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...), lang: str = Form('es')):
    """Transcribe audio a texto usando Whisper."""
    try:
        text = transcribe_audio(audio, lang)
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/tts")
async def tts_endpoint(text: str = Form(...), lang: str = Form('es')):
    """Convierte texto a audio usando pyttsx3."""
    try:
        audio_path = synthesize_text(text, lang)
        return FileResponse(audio_path, media_type='audio/wav', filename='tts.wav')
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Servir archivos estáticos (frontend webchat)
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')

# Ruta raíz: servir index.html
@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, 'index.html'))

# --- Policy Engine y Proxy de Seguridad ---
FORBIDDEN_PATTERNS = [
    r"(?i)ignore previous instructions",
    r"(?i)forget all previous",
    r"(?i)system prompt",
    r"(?i)you are now",
    r"(?i)act as",
    r"(?i)\bshutdown\b",
    r"(?i)\bdelete\b",
    r"(?i)\bdrop\b",
    r"(?i)\bexec\b",
    r"(?i)\bimport os\b",
    r"(?i)\bopen\b"
]

def sanitize_input(text: str) -> str:
    # Limpieza profunda: normaliza unicode, elimina caracteres peligrosos, espacios, y stopwords
    text = normalize_unicode(text)
    text = re.sub(r'[<>"\'`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = clean_text(text)
    return text

def policy_check(text: str) -> bool:
    # Rechaza patrones peligrosos y entradas vacías o triviales
    if not text or len(text.strip()) < 2:
        return False
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            return False
    return True

async def secure_proxy(request: Request, field: str = "answer"):
    data = await request.json()
    value = data.get(field, "")
    if not policy_check(value):
        raise HTTPException(status_code=400, detail="Input bloqueado por política de seguridad.")
    data[field] = sanitize_input(value)
    return data

# --- Endpoints principales ---
from fastapi import Body

@app.post("/start_interview")
async def start_interview(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    role = payload.get("role")
    interview_type = payload.get("type")
    mode = payload.get("mode", "practice")
    user_id = str(user_id or "")
    if not user_id:
        return {"error": "user_id is required"}
    return agent.start_interview(user_id, role, interview_type, mode)

@app.post("/next_question")
async def next_question(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    user_id = str(user_id or "")
    if not user_id:
        return {"error": "user_id is required"}
    return agent.next_question(user_id)

@app.post("/answer")
async def answer(request: Request):
    data = await secure_proxy(request, field="answer")
    user_id = data.get("user_id")
    answer = data.get("answer")
    # Procesar respuesta y actualizar contexto si es necesario
    session = agent.sessions.get(user_id)
    if session and session["stage"] == "context":
        # NLP simple: guardar respuesta en el contexto adecuado
        idx = session["context_asked"]
        if idx == 0:
            session["role"] = answer
        elif idx == 1:
            session["level"] = answer
        elif idx == 2:
            try:
                session["years"] = int(answer)
            except Exception:
                session["years"] = answer
        elif idx == 3:
            session["knowledge"] = [k.strip() for k in answer.split(",") if k.strip()]
        elif idx == 4:
            session["tools"] = [t.strip() for t in answer.split(",") if t.strip()]
        # Las demás respuestas se ignoran o pueden guardarse como notas
        # Avanzar a la siguiente pregunta de contexto o a la entrevista
        next_q = agent.next_question(user_id)
        if "question" in next_q:
            return {"feedback": "", "next": next_q["question"]}
        # Si ya terminó el contexto, iniciar la entrevista normal
        else:
            feedback = agent.process_answer(user_id, answer)
            return feedback
    else:
        feedback = agent.process_answer(user_id, answer)
        return feedback

@app.post("/end_interview")
async def end_interview(user_id: str):
    return agent.end_interview(user_id)

# --- Endpoints adicionales ---
@app.get("/get_roles")
def get_roles():
    """Lista todos los roles/cargos disponibles en el dataset técnico."""
    roles = set(q.get("role") for q in agent.tech_questions)
    roles = [role for role in roles if role is not None]
    return {"roles": sorted(roles)}

@app.get("/get_levels")
def get_levels():
    """Lista los niveles de experiencia soportados en el dataset técnico."""
    levels = set(q.get("level", "") for q in agent.tech_questions if "level" in q)
    return {"levels": sorted(list(levels))}

@app.get("/get_question_bank")
def get_question_bank(role: Optional[str] = None, level: Optional[str] = None):
    """Devuelve todas las preguntas filtradas por rol y/o nivel."""
    questions = agent.tech_questions
    if role:
        questions = [q for q in questions if q.get("role", "").lower() == role.lower()]
    if level:
        questions = [q for q in questions if q.get("level", "").lower() == level.lower()]
    return {"questions": questions}

@app.get("/interview_history")
def interview_history(user_id: str):
    """Consulta el historial de la entrevista en curso de un usuario."""
    session = agent.sessions.get(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="No hay entrevista activa para este usuario.")
    return {"history": session["history"]}

@app.post("/reset_interview")
def reset_interview(user_id: str):
    """Reinicia la sesión de entrevista de un usuario."""
    if user_id in agent.sessions:
        del agent.sessions[user_id]
    return {"message": "Entrevista reiniciada."}