
# Documentación Técnica: Arquitectura NLP, Embeddings, Clustering, Feedback Adaptativo y Aprendizaje Activo


## 1. Arquitectura General y Mejoras Recientes

Ready4Hire implementa una arquitectura modular y eficiente para simulación de entrevistas IA, integrando:
- **Embeddings semánticos** (SentenceTransformers, modelo `all-MiniLM-L6-v2` optimizado para CPU)
- **Selección estricta por rol/contexto**: solo se presentan preguntas relevantes al perfil usando embeddings y clustering, evitando preguntas fuera de dominio.
- **Clustering de desempeño**: las respuestas se agrupan por similitud para personalizar consejos, feedback y recursos.
- **Prompts adaptativos**: el LLM recibe contexto, historial y desempeño para generar feedback y pistas personalizadas.
- **Scoring robusto**: pondera similitud semántica, completitud, precisión y soft skills clave.
- **Feedback emocional y adaptativo**: análisis de emociones, motivación personalizada, aprendizaje de nuevas frases y emojis.
- **Memoria conversacional**: todo el historial de la sesión se usa para adaptar el flujo y el feedback.
- **Temporizador y control de inactividad**: engagement y cierre automático si el usuario no responde.
- **Aprendizaje activo y generación automática de preguntas**: el sistema aprende de cada sesión y expande el banco de preguntas y recursos.


### 1.1 Embeddings y Selección Semántica
Cada texto (pregunta, respuesta, contexto, rol) se convierte en un vector $\vec{x} \in \mathbb{R}^d$ usando SentenceTransformers. La similitud de coseno se usa para:
- Seleccionar solo preguntas relevantes al rol/contexto del usuario.
- Validar respuestas comparando con los campos `answer`/`expected` y buenas respuestas previas.
- Agrupar respuestas y preguntas por temas y desempeño (clustering).

---

## 2. Selección y Clustering de Preguntas


### 2.1 Perfilado y Filtrado Estricto
- Tras el contexto inicial, se genera un embedding de perfil.
- Solo se seleccionan preguntas con alta similitud al perfil (umbral configurable), evitando preguntas fuera de dominio.


### 2.2 Clustering de Desempeño y Consejos Personalizados
- Las respuestas del usuario se agrupan por similitud (KMeans u otro clustering eficiente).
- Cada cluster tiene plantillas de consejos y recursos personalizados.
- El feedback y los recursos se adaptan dinámicamente al grupo de desempeño del usuario.

---

## 3. Evaluación de Respuestas y Feedback Adaptativo


### 3.1 Proceso de Evaluación y Scoring
1. **Embeddings:** Cada respuesta se embebe y se compara con la esperada y buenas respuestas previas.
2. **Scoring robusto:** Se pondera similitud, completitud, precisión y soft skills clave.
3. **Clustering:** El usuario recibe consejos y recursos personalizados según su grupo de desempeño.
4. **Intentos y pistas:** Cada intento y uso de pista ajusta el puntaje y el feedback.
5. **Feedback emocional:** Se analiza la emoción predominante y se adapta el feedback y la motivación.
6. **Prompts adaptativos:** El LLM recibe contexto, historial y desempeño para generar feedback y pistas personalizadas.
7. **Sugerencias de recursos:** Si el usuario falla varias veces en un tema, se recomiendan recursos y explicaciones chain-of-thought.

### 3.2 Ejemplo de Evaluación
- Respuesta: "Es para crear contenedores."
- Respuesta esperada: "Permite crear, desplegar y gestionar contenedores de aplicaciones."
- Similitud coseno: 0.81 → Correcta.
- Cobertura de conceptos: 2/3 → Correcta.
- Feedback: "¡Respuesta correcta! 🚀 Suma un logro más a tu carrera. Puntaje: 8 (intentos: 2, penalización por pistas: 2)"

---


## 4. Ajuste Dinámico de Dificultad y Eficiencia

- El sistema ajusta la dificultad y el tipo de feedback según la racha de aciertos/errores y el cluster de desempeño.
- Se optimiza el uso de recursos (CPU, memoria) limitando hilos y usando modelos ligeros.
- El ajuste se realiza por reglas y ML, y se adapta a la infraestructura disponible.

---


## 5. Feedback Emocional, Motivacional y Adaptativo

- El feedback se adapta a la emoción detectada y al historial del usuario.
- El sistema aprende nuevas frases motivacionales y emojis usando LLMs y feedback de usuarios.
- El feedback y las pistas se generan con prompts adaptativos y explicaciones chain-of-thought.

---


## 6. Aprendizaje Activo, Memoria Conversacional y Generación Automática

- Todo el historial de la sesión se almacena y se usa para adaptar el flujo y el feedback.
- Las buenas respuestas y errores frecuentes se almacenan y refrescan los embeddings.
- El sistema utiliza LLMs para crear nuevas preguntas y recursos tras cada buena respuesta.
- Las nuevas preguntas se integran automáticamente al dataset y a los embeddings.

---

## 7. Métricas, Analítica y Panel de Administración

- Se calculan aciertos, errores y porcentajes de éxito por tipo de pregunta y por tema (clustering).
- Se identifican habilidades destacadas y a mejorar según las respuestas y el clustering de errores.
- Se mide el tiempo de respuesta promedio y se reconoce la agilidad o se sugieren técnicas de estudio si es necesario.
- El resumen final es generado por LLM, incluye estadísticas, fortalezas, puntos de mejora, advertencias, clustering de temas, recursos recomendados y reconocimiento por agilidad.
- El panel admin permite analizar desempeño, errores frecuentes, progreso y exportar resultados.

---


## 8. Integración de Audio, Temporizador y Webchat

- El frontend permite responder por texto o voz (STT) y escuchar las respuestas del agente (TTS).
- El backend expone endpoints `/stt` y `/tts` para integración de audio.
- El temporizador y la gestión de inactividad se implementan en frontend y backend para medir el tiempo y mantener el engagement.

---

## 9. Seguridad y Privacidad

- Todos los datos y métricas se almacenan de forma segura y anónima.
- El sistema cumple con buenas prácticas de seguridad, validación y protección de datos.

---


## 10. Extensibilidad y Futuro

- Arquitectura modular: permite agregar nuevos modelos, algoritmos ML, recursos y visualizaciones fácilmente.
- El feedback, la selección de preguntas y los recursos se pueden mejorar con modelos más avanzados, datos adicionales y nuevas estrategias de clustering y personalización.

---

**Última actualización: agosto 2025**
