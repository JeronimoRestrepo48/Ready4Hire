# Guía de Consumo del Backend Ready4Hire desde Webchat (2025)

## 1. Autenticación y Seguridad
- Todos los endpoints (excepto `/stt` y `/tts`) requieren JWT en el header `Authorization: Bearer <token>`.
- El token se obtiene tras el login (no cubierto aquí).
- CORS, HTTPS y rate limiting activos.

## 2. Flujo de Entrevista Interactiva

### a) Iniciar entrevista
```js
await fetch('/start_interview', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ...', 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id, role, level, mode })
});
```
- Respuesta: pregunta inicial y contexto.

### b) Responder pregunta
```js
await fetch('/answer', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ...', 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id, answer })
});
```
- Respuesta: feedback adaptativo, penalización por intentos/pistas, badges, puntos, si puede avanzar (`retry: false`), o si debe mejorar la respuesta (`retry: true`).
- El frontend debe bloquear el avance hasta que `retry` sea `false`.

### c) Siguiente pregunta
```js
await fetch('/next_question', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ...', 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id })
});
```
- Solo se debe llamar si la respuesta anterior fue correcta.

### d) Finalizar entrevista
```js
await fetch('/end_interview', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ...', 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id, satisfaction })
});
```
- Respuesta: resumen, temas débiles, recursos, badges, score, analítica avanzada.

### e) Encuesta de satisfacción
```js
await fetch('/survey', {
  method: 'POST',
  headers: { 'Authorization': 'Bearer ...', 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id, rating, comments })
});
```

## 3. STT y TTS (voz)

### a) Audio a texto (STT)
```js
const form = new FormData();
form.append('audio', audioBlob);
form.append('lang', 'es');
await fetch('/stt', { method: 'POST', body: form });
```
- Respuesta: `{ text: "..." }`

### b) Texto a audio (TTS)
```js
const form = new FormData();
form.append('text', 'Respuesta de ejemplo');
form.append('lang', 'es');
await fetch('/tts', { method: 'POST', body: form });
```
- Respuesta: archivo de audio (WAV/MP3)

## 4. Otras utilidades
- `/get_roles` y `/get_levels`: para poblar selects de rol y nivel.
- `/get_question_bank`: para mostrar banco de preguntas filtradas.
- `/interview_history`: para mostrar historial y analítica en tiempo real.
- `/reset_interview`: para reiniciar la entrevista.

## 5. Buenas prácticas de integración
- Siempre mostrar feedback antes de permitir avanzar.
- Penalizar puntos y mostrar badges según intentos y uso de pistas.
- Mostrar contador visual de preguntas y barra de progreso.
- Registrar y mostrar feedback final, temas débiles y recursos sugeridos.
- Usar STT/TTS para accesibilidad y experiencia inmersiva.
- Manejar errores y expiración de JWT de forma elegante.

---
Última actualización: agosto 2025
