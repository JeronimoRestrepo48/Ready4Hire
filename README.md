# Ready4Hire - Plataforma de Simulación de Entrevistas con IA Adaptativa, Analítica y Gamificación


## Descripción General
Ready4Hire es una plataforma avanzada para la simulación de entrevistas técnicas y de soft skills, orientada a ingeniería de sistemas y tecnología. Integra inteligencia artificial, machine learning, clustering de temas, ajuste dinámico de dificultad, gamificación, feedback emocional y adaptativo, analítica avanzada y un panel de administración robusto. Permite a los usuarios practicar entrevistas, recibir retroalimentación personalizada y didáctica, visualizar su progreso y a los administradores analizar el desempeño global y por usuario.

### Tecnologías y Arquitectura
- **Backend:** FastAPI, Python 3.10+, SentenceTransformers, Ollama (LLM), clustering, feedback loop ML, ajuste dinámico de dificultad, analítica avanzada, PostgreSQL.
- **Frontend:** Webchat moderno (HTML/JS/CSS), integración STT (voz a texto) y TTS (texto a voz), Chart.js, Bootstrap 5, accesibilidad y seguridad.
- **NLP/ML:** Embeddings, similitud semántica, clustering de temas, análisis emocional (transformers), aprendizaje activo, generación automática de preguntas, feedback adaptativo y motivacional.
- **Seguridad:** JWT, CORS, HTTPS, rate limiting, validación robusta, protección XSS/CSRF, roles y permisos.

## Características Clave
- **Flujo de entrevista interactivo:** Alterna preguntas técnicas y blandas personalizadas usando embeddings, clustering y contexto del usuario.
- **Feedback inmediato, emocional y adaptativo:** Cada respuesta es evaluada matemáticamente (similitud semántica, cobertura de conceptos, intentos, uso de pistas, emociones) y el feedback es motivador, emocional y didáctico. El feedback se adapta a patrones de error, emociones y progreso.
- **Bloqueo de avance:** El usuario no puede avanzar a la siguiente pregunta hasta responder correctamente la actual. El feedback se muestra siempre antes de la siguiente pregunta.
- **Gamificación avanzada:** Puntos decrecientes por intentos, penalización por uso de pistas, badges, niveles, rachas y reconocimiento por agilidad. El feedback final penaliza el rendimiento si hubo muchos intentos o pistas.
- **Ajuste dinámico de dificultad:** El sistema sube o baja la dificultad según el desempeño y la racha del usuario.
- **Clustering de temas y sugerencias de recursos:** El sistema detecta temas dominados y débiles, recomienda rutas de aprendizaje y recursos personalizados.
- **Aprendizaje activo:** El sistema aprende de buenas respuestas, genera nuevas preguntas y ajusta la selección de preguntas y feedback.
- **Integración de audio:** Soporte robusto para STT (voz a texto) y TTS (texto a voz) en frontend y backend.
- **Analítica avanzada:** Métricas de desempeño, tiempo de respuesta, clustering de errores, panel de administración y filtros avanzados.
- **Frontend webchat accesible:** Chat moderno, visualización de feedback, controles de audio, temporizador, experiencia inclusiva y segura.

## Flujo de Entrevista y Feedback (Frontend + Backend)

1. **Inicio:** El usuario selecciona rol, nivel y modo (práctica/examen). El backend personaliza el pool de preguntas usando embeddings, clustering y contexto.
2. **Pregunta:** El agente muestra una pregunta personalizada (técnica o soft skill).
3. **Respuesta:** El usuario responde por texto o voz. El backend evalúa la respuesta:
   - Calcula similitud semántica, cobertura de conceptos clave y emociones.
   - Si la respuesta es suficientemente correcta, muestra feedback positivo y permite avanzar.
   - Si no, muestra feedback didáctico, emocional, sugerencias de recursos y bloquea el avance hasta que la respuesta sea adecuada.
   - Penaliza los puntos por cada intento adicional y por uso de pistas/sugerencias.
   - El feedback se imprime en consola y se muestra en el chat antes de la siguiente pregunta.
4. **Gamificación:** Los puntos, nivel y badges se actualizan dinámicamente. Menos puntos por más intentos y pistas. Reconocimiento por agilidad y mejora en temas.
5. **Finalización:** Al terminar, el backend genera un resumen penalizado si hubo muchos intentos o pistas, mostrando advertencias, clustering de temas, recursos recomendados y reconocimiento en el feedback final.
6. **Encuesta y analítica:** El usuario puede dejar feedback y ver su progreso en el dashboard.

## Algoritmo de Evaluación de Respuestas
- Similitud semántica (coseno) > 0.75 (técnico) o > 0.65 (soft) = correcta.
- Cobertura de conceptos clave (palabras clave) > 60% = correcta.
- Cada intento adicional resta puntos (-2 por intento extra).
- Uso de pistas/sugerencias resta puntos (-2 por pista).
- El avance está bloqueado hasta que la respuesta sea correcta.
- El feedback siempre se muestra antes de la siguiente pregunta.
- El feedback se adapta a emociones, patrones de error y progreso.

## Penalización y Feedback Final
- Si el usuario usó muchas pistas o necesitó muchos intentos, el feedback final incluye advertencias y penaliza el rendimiento global.
- El resumen final es generado por LLM, incluye estadísticas, fortalezas, puntos de mejora, advertencias, clustering de temas, recursos recomendados y reconocimiento por agilidad.

## Documentación Técnica y API

- [Documentación de API y Seguridad](docs/API_DOCUMENTACION.md)
- [Consumo desde Backend Webchat](docs/CONSUMO_BACKEND_WEBCHAT.md)
- [Motor NLP, Embeddings y Feedback Adaptativo](docs/NLP_EMBEDDINGS_TECH.md)

## Fundamentos Matemáticos y ML

- Embeddings de texto con SentenceTransformers (`all-MiniLM-L6-v2`).
- Similitud semántica: coseno entre embeddings.
- Clustering de temas y errores frecuentes.
- Feedback emocional: análisis de emociones con transformers.
- Aprendizaje automático: el sistema aprende de buenas respuestas y genera nuevas preguntas.
- Ver detalles en [docs/NLP_EMBEDDINGS_TECH.md](docs/NLP_EMBEDDINGS_TECH.md).

## Integración Frontend-Backend

- El frontend (chat.js) solo permite avanzar si el feedback recibido no tiene `retry`.
- El feedback se muestra siempre antes de la siguiente pregunta.
- El feedback final penalizado y adaptativo se muestra completo en el chat.
- El backend expone endpoints robustos para todo el flujo, con validación, seguridad y logging.

## Estructura del Proyecto

```
app/
  ├── main.py                # Entrypoint FastAPI, rutas principales, integración ML
  ├── models.py              # Modelos SQLAlchemy: User, Interview, Achievement, UserPDF
  ├── database.py            # Configuración y helpers de base de datos (async, PostgreSQL)
  ├── auth.py                # Lógica de autenticación JWT, registro y login
  ├── user_routes.py         # Endpoints de usuario, dashboard, admin, analítica, PDF
  ├── interview_agent.py     # Lógica de simulación de entrevistas, feedback, LLM
  ├── ml_recommendations.py  # Algoritmos de recomendación y clustering
  ├── ml_feedback_loop.py    # Analítica y feedback loop para admins
  ├── ml_dynamic_difficulty.py # Ajuste dinámico de dificultad
  └── static/
      ├── dashboard.html/js/css  # Frontend usuario
      ├── profile.html/js        # Perfil usuario
      ├── admin.html/js          # Panel admin, analítica, filtros, gráficas
      ├── chat.js                # Chat y feedback
      └── chartjs-loader.js      # Loader de Chart.js para gráficas
```

## Instalación y Configuración

1. **Requisitos**:
   - Python 3.10+
   - PostgreSQL
   - Node.js (solo si deseas modificar frontend avanzado)
   - Paquetes: ver `requirements.txt`

2. **Variables de entorno**:
   - `DATABASE_URL`: URL de conexión PostgreSQL
   - `SECRET_KEY`: clave secreta JWT

3. **Inicialización**:
   - Instala dependencias: `pip install -r requirements.txt`
   - Crea la base de datos y ejecuta migraciones si aplica
   - Ejecuta el backend: `uvicorn app.main:app --reload`
   - Accede a `/static/login.html` para iniciar sesión

## Seguridad
- JWT para autenticación
- Endpoints protegidos por permisos (usuario/admin)
- Validación robusta de datos y errores
- Rate limiting y protección DoS
- Validación de archivos y payloads
- CORS estricto y HTTPS/HSTS
- Headers seguros y cookies protegidas
- Sanitización y escape de entradas en frontend y backend
- Logout seguro y protección clickjacking

## Extensibilidad
- Modularidad para añadir nuevos algoritmos ML, visualizaciones, endpoints o vistas.
- Fácil de agregar nuevos modelos, recursos y analítica.

## Audio y Webchat
- Responder por texto o voz (STT).
- Escuchar las respuestas del agente (TTS).
- Visualizar el temporizador en modo examen.
- Cambiar entre modo práctica y examen.
- Interfaz moderna, responsiva, accesible y segura.

## Aprendizaje Automático y Deep Learning
- `app/emotion_analyzer.py`: Análisis de emociones en texto usando modelos transformers de HuggingFace.
- Feedback adaptativo: el agente ajusta su respuesta y motivación según la emoción detectada en el usuario.
- Aprendizaje automático de frases motivacionales y emojis: el sistema expande dinámicamente su repertorio usando LLMs.
- Generación automática de nuevas preguntas técnicas: el agente crea preguntas inéditas y las integra al dataset y embeddings.

## Analítica y Panel de Administración
- Métricas de desempeño, clustering de temas, errores frecuentes, tiempo de respuesta, satisfacción, progreso y logros.
- Panel admin con filtros avanzados, gráficas y exportación de resultados.

---
Última actualización: agosto 2025
