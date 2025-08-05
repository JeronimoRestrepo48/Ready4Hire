
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.interview_agent import InterviewAgent
from app.core import utils

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app.mount('/static', StaticFiles(directory=static_dir), name='static')

agent = InterviewAgent()
user_last_feedback = {}

# Endpoint raíz: sirve la interfaz webchat
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Iniciar entrevista: tipo, modo, preguntas de contexto
@app.post("/start_interview")
async def start_interview(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    interview_type = data.get("interview_type", "tecnica")
    mode = data.get("mode", "practica")
    context_questions = agent.start_interview(user_id, interview_type, mode)
    if isinstance(context_questions, dict) and "question" in context_questions:
        return {"context_questions": context_questions}
    elif isinstance(context_questions, str):
        return {"context_questions": {"question": context_questions}}
    else:
        return {"context_questions": {"question": "Bienvenido. ¿Para qué rol deseas prepararte?"}}

# Recibir respuesta de contexto y avanzar
@app.post("/context_answer")
async def context_answer(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    answer = data.get("answer", "")
    return agent.process_context_answer(user_id, answer)

# Siguiente pregunta de entrevista con feedback y contador
@app.post("/next_question")
async def next_question(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    session = agent.sessions.get(user_id)
    if not session:
        return {"error": "Entrevista no iniciada"}
    # Si aún en contexto, solo devolver la siguiente pregunta de contexto
    if session.get("stage") == "context":
        result = agent.next_question(user_id)
        return {"question": result["question"] if isinstance(result, dict) and "question" in result else result}
    # Si terminó la entrevista (10 preguntas), llamar end_interview y devolver feedback final y trigger de survey
    if session.get('stage') == 'interview' and session.get('question_counter', 0) >= 10:
        feedback = agent.end_interview(user_id)
        return {"end": True, "final_feedback": feedback, "survey": True}
    # Feedback de la última respuesta
    feedback = None
    hist = session.get("history", [])
    for h in reversed(hist):
        if "feedback" in h:
            feedback = h["feedback"]
            break
    # Siguiente pregunta
    # Incrementar el contador ANTES de devolver la pregunta
    if session.get('stage') == 'interview':
        session['question_counter'] = session.get('question_counter', 0) + 1
    result = agent.next_question(user_id)
    resp = {"question": result["question"] if isinstance(result, dict) and "question" in result else result,
            "counter": session.get('question_counter', 0),
            "feedback": feedback}
    return resp

# Responder pregunta de entrevista
@app.post("/answer")
async def answer(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    answer = data.get("answer", "")
    if not answer.strip():
        return {"error": "La respuesta no puede estar vacía."}
    feedback = agent.process_answer(user_id, answer)
    user_last_feedback[user_id] = feedback
    return feedback


## El endpoint /feedback ha sido eliminado. El feedback se entrega ahora en /next_question.

# Terminar entrevista y feedback global
@app.post("/end_interview")
async def end_interview(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    feedback = agent.end_interview(user_id)
    return {"feedback": feedback}

# Recibir encuesta de satisfacción
@app.post("/survey")
async def survey(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "test")
    rating = data.get("rating", 0)
    comments = data.get("comments", "")
    agent.process_survey(user_id, rating, comments)
    return {"message": "¡Gracias por tu feedback!"}


# STT: Speech to Text
@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    # Aquí se integraría el modelo de STT real
    return {"text": "(transcripción simulada)"}

# TTS: Text to Speech
@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text", "")
    # Aquí se integraría el modelo de TTS real
    return {"audio_url": "/static/fake_audio.wav"}

