// --- STT y TTS integrados con backend ---
const micBtn = document.getElementById('mic-btn');
const ttsBtn = document.getElementById('tts-btn');
let lastAgentMsg = '';
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

if (micBtn) {
  micBtn.onclick = async () => {
    if (isRecording) {
      mediaRecorder.stop();
      micBtn.innerHTML = '<span>🎤</span>';
      isRecording = false;
      return;
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      alert('Tu navegador no soporta grabación de audio.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');
        formData.append('lang', 'es');
        const res = await fetch('/stt', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.text) {
          userInput.value = data.text;
          userInput.focus();
        } else if (data.error) {
          alert('Error STT: ' + data.error);
        }
      };
      mediaRecorder.start();
      micBtn.innerHTML = '<span>⏹️</span>';
      isRecording = true;
    } catch (err) {
      alert('No se pudo acceder al micrófono: ' + err);
    }
  };
}

if (ttsBtn) {
  ttsBtn.onclick = async () => {
    if (!lastAgentMsg) return;
    const formData = new FormData();
    formData.append('text', lastAgentMsg);
    formData.append('lang', 'es');
    const res = await fetch('/tts', { method: 'POST', body: formData });
    if (res.ok) {
      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } else {
      alert('Error TTS');
    }
  };
}
const API_URL = "";
let userId = "user-" + Math.random().toString(36).substring(2, 10);
let interviewType = "technical";
let mode = "practice";
let timerInterval = null;
let timerStart = null;
let started = false;
let questionCount = 0;
const MAX_QUESTIONS = 10;

const chatBox = document.getElementById('chat-box');
const timerDiv = document.getElementById('timer');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const startBtn = document.getElementById('start-btn');
const setupSection = document.getElementById('setup-section');
const chatSection = document.getElementById('chat-section');

const interviewTypeSelect = document.getElementById('interview-type');
const modeSelect = document.getElementById('mode-select');


startBtn.onclick = async () => {
  interviewType = interviewTypeSelect.value;
  mode = modeSelect.value;
  setupSection.style.display = 'none';
  chatSection.style.display = '';
  started = true;
  questionCount = 0;
  if (mode === 'exam') {
    timerDiv.style.display = '';
    timerStart = Date.now();
    timerDiv.textContent = '00:00';
    timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - timerStart) / 1000);
      const min = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const sec = String(elapsed % 60).padStart(2, '0');
      timerDiv.textContent = `${min}:${sec}`;
    }, 1000);
  } else {
    timerDiv.style.display = 'none';
    if (timerInterval) clearInterval(timerInterval);
  }
  await startInterview();
};


function sanitizeInput(text) {
  // Elimina caracteres peligrosos, espacios excesivos y normaliza unicode básico
  text = text.replace(/[<>"'`]/g, '');
  text = text.replace(/\s+/g, ' ').trim();
  // Opcional: quitar tildes
  text = text.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  return text;
}

chatForm.onsubmit = async (e) => {
  e.preventDefault();
  let text = userInput.value.trim();
  if (!text) return;
  text = sanitizeInput(text);
  addMessage('user', text);
  userInput.value = '';
  await sendAnswer(text);
};

function addMessage(sender, text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message ' + sender;
  // Detectar bloque de gamificación y resaltarlo
  if (sender === 'agent' && text.includes('🎮 Sistema de Gamificación Avanzada 🎮')) {
    // Separar resumen y gamificación
    const [summary, gamification] = text.split('🎮 Sistema de Gamificación Avanzada 🎮');
    if (summary.trim()) {
      const summaryDiv = document.createElement('div');
      summaryDiv.className = 'bubble ' + sender + ' msg-summary';
      summaryDiv.innerText = summary.trim();
      msgDiv.appendChild(summaryDiv);
    }
    const gamDiv = document.createElement('div');
    gamDiv.className = 'bubble ' + sender + ' msg-gamification';
    gamDiv.innerHTML = `<span style="font-size:1.2em;">🎮 <b>Sistema de Gamificación Avanzada</b> 🎮</span><pre style="white-space:pre-wrap;font-family:inherit;margin:0;">${gamification.trim()}</pre>`;
    msgDiv.appendChild(gamDiv);
  } else {
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + sender;
    bubble.innerText = text;
    msgDiv.appendChild(bubble);
  }
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  if (sender === 'agent') lastAgentMsg = text;
}
// Estilos para gamificación (puedes mover esto a tu CSS principal)
const style = document.createElement('style');
style.innerHTML = `
.msg-gamification {
  background: #232b4d;
  color: #fff;
  border: 2px solid #ffd700;
  border-radius: 10px;
  margin: 10px 0 0 0;
  padding: 12px 16px;
  font-size: 1.08em;
  box-shadow: 0 2px 8px #0002;
}
.msg-summary {
  margin-bottom: 8px;
}
`;
document.head.appendChild(style);

async function startInterview() {
  const res = await fetch('/start_interview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, type: interviewType, mode })
  });
  const data = await res.json();
  if (data.question) addMessage('agent', data.question);
}

async function sendAnswer(answer) {
  questionCount++;
  const res = await fetch('/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, answer })
  });
  const data = await res.json();
  if (data.next) {
    addMessage('agent', data.next);
    return;
  }
  if (data.feedback) addMessage('agent', data.feedback);
  if (questionCount < MAX_QUESTIONS) {
    await nextQuestion();
  } else {
    await endInterview();
  }
}

async function nextQuestion() {
  const res = await fetch('/next_question', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId })
  });
  const data = await res.json();
  if (data.question) addMessage('agent', data.question);
}

async function endInterview() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerDiv.style.display = 'none';
  const res = await fetch('/end_interview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId })
  });
  const data = await res.json();
  if (data.summary) addMessage('agent', data.summary);
  userInput.disabled = true;
}
