// --- Lógica completa de entrevista interactiva con IA ---
// Elementos UI
const setupSection = document.getElementById('setup-section');
const contextSection = document.getElementById('context-section');
const mainChatSection = document.getElementById('main-chat-section');
const feedbackSection = document.getElementById('feedback-section');
const contextQuestionDiv = document.getElementById('context-question');
const contextForm = document.getElementById('context-form');
const contextInput = document.getElementById('context-input');
const chatBox = document.getElementById('chat-box');
const questionCounterDiv = document.getElementById('question-counter');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const gamificationBar = document.getElementById('gamification-bar');
const pointsSpan = document.getElementById('points');
const levelSpan = document.getElementById('level');
const startBtn = document.getElementById('start-btn');
const interviewType = document.getElementById('interview-type');
const interviewMode = document.getElementById('interview-mode');
const micBtn = document.getElementById('mic-btn');
const ttsBtn = document.getElementById('tts-btn');
const finalFeedbackDiv = document.getElementById('final-feedback');
const surveyForm = document.getElementById('survey-form');
const surveyRating = document.getElementById('survey-rating');
const surveyComments = document.getElementById('survey-comments');
const surveyThanks = document.getElementById('survey-thanks');


let userId = 'user-' + Math.random().toString(36).substring(2, 10);
let currentPoints = 0;
let currentLevel = 1;
let contextDone = false;
let questionCount = 0;
const maxQuestions = 10;
let inInterview = false;

// Mensaje de carga y alertas
const loadingDiv = document.createElement('div');
loadingDiv.id = 'loading-msg';
loadingDiv.style.display = 'none';
loadingDiv.style.textAlign = 'center';
loadingDiv.style.margin = '10px 0';
loadingDiv.innerHTML = '<span>⏳ Procesando...</span>';
chatBox.parentNode.insertBefore(loadingDiv, chatBox);

function showLoading(show = true) {
  loadingDiv.style.display = show ? '' : 'none';
}

function showSection(section) {
  [setupSection, contextSection, mainChatSection, feedbackSection].forEach(s => s.classList.remove('active'));
  section.classList.add('active');
}

function appendMessage(msg, sender) {
  const div = document.createElement('div');
  if (sender === 'user') {
    div.className = 'msg-user';
    div.innerHTML = msg;
  } else {
    div.className = 'msg-agent';
    // Detectar si es pregunta de entrevista (no contexto)
    const isQuestion = msg.trim().endsWith('?') || /^\d+\.\s/.test(msg.trim()) || (msg.trim().toLowerCase().includes('¿qué') || msg.trim().toLowerCase().includes('¿cuál'));
    if (isQuestion && inInterview) {
      // Evita doble enumeración
      let content = msg;
      if (!/^\d+\.\s/.test(msg.trim())) {
        questionCount++;
        content = `${questionCount}. ${msg}`;
      }
      div.innerHTML = `<span style=\"color:#1a5cff;font-weight:bold;\">${content}</span>`;
      if (questionCounterDiv) {
        questionCounterDiv.textContent = `Pregunta ${questionCount} / ${maxQuestions}`;
        questionCounterDiv.style.display = '';
      }
    } else {
      div.innerHTML = msg;
    }
  }
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showError(msg) {
  appendMessage(`<span style="color:#c00;font-weight:bold;">${msg}</span>`, 'agent');
}

// --- Inicio: selección de tipo y modo ---
startBtn.onclick = async () => {
  showSection(contextSection);
  showLoading(true);
  try {
    const res = await fetch('/start_interview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        interview_type: interviewType.value,
        mode: interviewMode.value
      })
    });
    const data = await res.json();
    if (data.context_questions && data.context_questions.question) {
      contextQuestionDiv.textContent = data.context_questions.question;
    } else {
      showError('No se pudo iniciar la entrevista.');
    }
  } catch (err) {
    showError('Error de red al iniciar entrevista.');
  }
  showLoading(false);
  contextInput.value = '';
  contextInput.focus();
};

// --- Preguntas de contexto ---
contextForm.onsubmit = async (e) => {
  e.preventDefault();
  const answer = contextInput.value.trim();
  if (!answer) return;
  showLoading(true);
  try {
    const res = await fetch('/context_answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, answer })
    });
    const data = await res.json();
    if (data.question) {
      contextQuestionDiv.textContent = data.question;
      contextInput.value = '';
      contextInput.focus();
    } else {
      // Contexto finalizado
      showSection(mainChatSection);
      contextDone = true;
      chatBox.innerHTML = '';
      questionCount = 0;
      inInterview = true;
      questionCounterDiv.textContent = `Pregunta 0 / ${maxQuestions}`;
      questionCounterDiv.style.display = '';
      await nextInterviewQuestion();
    }
  } catch (err) {
    showError('Error de red al enviar respuesta de contexto.');
  }
  showLoading(false);
};


// --- Chat principal: responder preguntas de entrevista ---
chatForm.onsubmit = async (e) => {
  e.preventDefault();
  const msg = userInput.value.trim();
  if (!msg) return;
  appendMessage(msg, 'user');
  userInput.value = '';
  showLoading(true);
  try {
    // Enviar respuesta al backend y obtener feedback inmediato
    const res = await fetch('/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, answer: msg })
    });
    const feedbackData = await res.json();
    if (feedbackData && feedbackData.feedback) {
      appendMessage(feedbackData.feedback, 'agent');
    } else if (feedbackData && feedbackData.motivation) {
      appendMessage(feedbackData.motivation, 'agent');
    } else if (feedbackData && feedbackData.summary) {
      appendMessage(`<b>Resumen IA:</b> ${feedbackData.summary}`, 'agent');
    } else if (feedbackData && feedbackData.error) {
      showError(feedbackData.error);
    }
    // Si la respuesta NO es correcta, no avanzar pregunta
    if (feedbackData && feedbackData.retry) {
      showLoading(false);
      return;
    }
    // Si la respuesta fue correcta, pedir la siguiente pregunta (y feedback) al backend
    await nextInterviewQuestion();
  } catch (err) {
    showError('Error de red al enviar respuesta o al obtener feedback.');
  }
  showLoading(false);
};

// --- TTS: Text to Speech ---
ttsBtn.addEventListener('click', async () => {
  // Busca la última pregunta del agente
  const agentMsgs = Array.from(chatBox.querySelectorAll('.msg-agent span, .msg-agent'));
  let lastQ = '';
  for (let i = agentMsgs.length - 1; i >= 0; i--) {
    const txt = agentMsgs[i].textContent || '';
    if (txt.endsWith('?')) { lastQ = txt.replace(/^\d+\.\s*/, ''); break; }
  }
  if (!lastQ) return;
  const res = await fetch('/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: lastQ })
  });
  const data = await res.json();
  if (data.audio_url) {
    const audio = new Audio(data.audio_url);
    audio.play();
  }
});


chatForm.onsubmit = async (e) => {
  e.preventDefault();
  const msg = userInput.value.trim();
  if (!msg) return;
  appendMessage(msg, 'user');
  userInput.value = '';
  showLoading(true);
  try {
    // Enviar respuesta al backend y obtener feedback inmediato
    const res = await fetch('/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, answer: msg })
    });
    const feedbackData = await res.json();
    if (feedbackData && feedbackData.feedback) {
      appendMessage(feedbackData.feedback, 'agent');
    } else if (feedbackData && feedbackData.motivation) {
      appendMessage(feedbackData.motivation, 'agent');
    } else if (feedbackData && feedbackData.summary) {
      appendMessage(`<b>Resumen IA:</b> ${feedbackData.summary}`, 'agent');
    } else if (feedbackData && feedbackData.error) {
      showError(feedbackData.error);
    }
    // Si la respuesta NO es correcta, no avanzar pregunta
    if (feedbackData && feedbackData.retry) {
      showLoading(false);
      return;
    }
    // Si se llegó al máximo de preguntas, mostrar encuesta y feedback general
    if (feedbackData && (feedbackData.end || questionCount >= maxQuestions)) {
      showSurveyAndFeedback();
      inInterview = false;
      if (questionCounterDiv) {
        questionCounterDiv.textContent = '';
        questionCounterDiv.style.display = 'none';
      }
      showLoading(false);
      return;
    }
    if (feedbackData && feedbackData.points !== undefined) {
      currentPoints = feedbackData.points;
      pointsSpan.textContent = 'Puntos: ' + currentPoints;
      gamificationBar.classList.add('active');
    }
    if (feedbackData && feedbackData.level !== undefined) {
      currentLevel = feedbackData.level;
      levelSpan.textContent = 'Nivel: ' + currentLevel;
      gamificationBar.classList.add('active');
    }
    // Solo mostrar la siguiente pregunta si la respuesta fue correcta
    if (feedbackData && feedbackData.next && questionCount < maxQuestions) {
      appendMessage(feedbackData.next, 'agent');
    }
    if (feedbackData && feedbackData.end) {
      endInterview();
    }
  } catch (err) {
    showError('Error de red al enviar respuesta o al obtener feedback.');
  }
  showLoading(false);
};

async function nextInterviewQuestion() {
  showLoading(true);
  try {
    const res = await fetch('/next_question', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId })
    });
    const data = await res.json();
    if (data.end && data.final_feedback) {
      // Mostrar feedback final y encuesta
      showSection(feedbackSection);
      finalFeedbackDiv.innerHTML = `<b>Resumen IA:</b><br>${data.final_feedback.summary || ''}`;
      surveyForm.style.display = '';
      surveyThanks.style.display = 'none';
      inInterview = false;
      if (questionCounterDiv) {
        questionCounterDiv.textContent = '';
        questionCounterDiv.style.display = 'none';
      }
      showLoading(false);
      return;
    }
    // Mostrar feedback de la respuesta anterior (si existe)
    if (data.feedback) {
      appendMessage(data.feedback, 'agent');
    }
    // Mostrar la siguiente pregunta y actualizar contador
    if (data.question) {
      appendMessage(data.question, 'agent');
      if (typeof data.counter === 'number') {
        questionCount = data.counter;
        questionCounterDiv.textContent = `Pregunta ${questionCount} / ${maxQuestions}`;
        questionCounterDiv.style.display = '';
      }
    }
    if (data.error) showError(data.error);
  } catch (err) {
    showError('Error de red al obtener pregunta.');
  }
  showLoading(false);
}

// --- Terminar entrevista y mostrar feedback final ---
async function endInterview() {
  showSection(feedbackSection);
  showLoading(true);
  try {
    const res = await fetch('/end_interview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId })
    });
    const data = await res.json();
    if (data.feedback && data.feedback.summary) {
      finalFeedbackDiv.innerHTML = `<b>Resumen IA:</b><br>${data.feedback.summary}<br><b>Puntaje:</b> ${data.feedback.score} <br><b>Nivel:</b> ${data.feedback.level} <br><b>Puntos:</b> ${data.feedback.points}`;
    } else {
      showError('No se pudo obtener feedback final.');
    }
  } catch (err) {
    showError('Error de red al finalizar entrevista.');
  }
  showLoading(false);
}

// --- Encuesta de satisfacción ---
surveyForm.onsubmit = async (e) => {
  e.preventDefault();
  const rating = parseInt(surveyRating.value);
  const comments = surveyComments.value.trim();
  showLoading(true);
  try {
    await fetch('/survey', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, rating, comments })
    });
    surveyForm.style.display = 'none';
    surveyThanks.classList.add('active');
  } catch (err) {
    showError('Error de red al enviar feedback.');
  }
  showLoading(false);
};

// --- STT: Speech to Text ---
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
if (micBtn) {
  micBtn.onclick = async () => {
    if (isRecording) {
      mediaRecorder.stop();
      micBtn.innerHTML = '🎤';
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
        if (audioBlob.size > 5 * 1024 * 1024) {
          alert('Archivo de audio demasiado grande (máx 5MB)');
          return;
        }
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
      micBtn.innerHTML = '⏹️';
      isRecording = true;
    } catch (err) {
      alert('No se pudo acceder al micrófono: ' + err);
    }
  };
}

// --- TTS: Text to Speech ---
if (ttsBtn) {
  ttsBtn.onclick = async () => {
    // Toma el último mensaje del agente
    const agentMsgs = chatBox.querySelectorAll('.msg-agent');
    if (!agentMsgs.length) return;
    const lastMsg = agentMsgs[agentMsgs.length - 1].textContent;
    const res = await fetch('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: lastMsg })
    });
    const data = await res.json();
    if (data.audio_url) {
      const audio = new Audio(data.audio_url);
      audio.play();
    }
  };
}
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
        // Validación de tamaño antes de enviar
        if (audioBlob.size > 5 * 1024 * 1024) {
          alert('Archivo de audio demasiado grande (máx 5MB)');
          return;
        }
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



async function nextQuestion() {
  const res = await fetch('/next_question', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (localStorage.getItem('access_token')||'') },
    body: JSON.stringify({ user_id: userId })
  });
  if (res.status === 401) {
    showAlert('Sesión expirada. Por favor inicia sesión de nuevo.', 'warning');
    setTimeout(()=>window.location.href='/static/login.html', 2000);
    return;
  }
  if (!res.ok) {
    showAlert('Error de red o servidor. Intenta de nuevo.', 'danger');
    return;
  }
  const data = await res.json();
  if (data.question) addMessage('agent', data.question);
}



async function endInterview() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerDiv.classList.add('timer-hidden');
  let res;
  try {
    res = await fetch('/end_interview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (localStorage.getItem('access_token')||'') },
      body: JSON.stringify({ user_id: userId })
    });
    if (res.status === 401) {
      showAlert('Sesión expirada. Por favor inicia sesión de nuevo.', 'warning');
      setTimeout(()=>window.location.href='/static/login.html', 2000);
      return;
    }
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    // Feedback final largo, PDF y recomendaciones solo al final
    if (data.summary) {
      let feedbackHtml = `<div class="alert alert-success mt-3"><b>Feedback final:</b><br>${data.summary.replace(/\n/g, '<br>')}</div>`;
      if (data.pdf_path) {
        feedbackHtml += `<div class="mt-2"><b>Descarga tu reporte PDF:</b> <a href="${data.pdf_path}" target="_blank">Ver PDF</a></div>`;
      }
      if (data.learning_path && data.learning_path.length) {
        feedbackHtml += `<div class="mt-2"><b>Recomendaciones personalizadas:</b><ul>`;
        data.learning_path.forEach(link => {
          feedbackHtml += `<li><a href="${link}" target="_blank">${link}</a></li>`;
        });
        feedbackHtml += `</ul></div>`;
      }
      addMessage('agent', feedbackHtml);
    }
    userInput.disabled = true;
  } catch (err) {
    showAlert('Error de red o servidor. Intenta de nuevo.', 'danger');
  }
}

function showAlert(msg, type='danger') {
  alertDiv.innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">${msg}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Cerrar"></button></div>`;
  alertDiv.style.display = '';
}

function hideAlert() {
  alertDiv.innerHTML = '';
  alertDiv.style.display = 'none';
}
}
