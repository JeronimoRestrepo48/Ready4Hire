# Documentación Técnica: Motor NLP, Embeddings, Clustering, Feedback Emocional y Aprendizaje Automático

## 1. Arquitectura General y Fundamentos Matemáticos

El sistema Ready4Hire utiliza una arquitectura híbrida de NLP y ML para personalizar, evaluar y mejorar la experiencia de entrevistas. Los componentes principales son:
- **Embeddings semánticos** (SentenceTransformers, modelo `all-MiniLM-L6-v2`)
- **Clustering y selección de preguntas** (por similitud y temas)
- **Feedback emocional y adaptativo** (análisis de emociones, motivación personalizada)
- **Ajuste dinámico de dificultad** (ML y reglas adaptativas)
- **Aprendizaje activo y generación automática de preguntas** (LLM, feedback loop)
- **Analítica avanzada y métricas** (desempeño, tiempo, errores, progreso)

### 1.1 Embeddings de Texto
Cada texto (pregunta, respuesta, contexto) se convierte en un vector $\vec{x} \in \mathbb{R}^d$:

- $\text{embedding}(\text{texto}) = \vec{x}$
- Se utiliza SentenceTransformers para obtener estos vectores.

**Similitud semántica:**
$$
\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}
$$
Un valor cercano a 1 indica alta similitud semántica.

---

## 2. Selección y Clustering de Preguntas

### 2.1 Perfilado del Usuario
- Tras el contexto inicial (rol, nivel, años, conocimientos, herramientas), se genera un embedding de perfil.
- Se calcula la similitud de coseno entre el embedding del perfil y todos los embeddings de preguntas técnicas y blandas.

### 2.2 Clustering Temático
- Las preguntas están etiquetadas por tema (`topic`).
- Se agrupan por similitud y temas dominados/débiles usando conteo de aciertos/errores y clustering simple.
- Se seleccionan las 10 preguntas más relevantes y balanceadas (técnicas y blandas) para cada usuario.
- El sistema alterna preguntas técnicas y blandas según preferencia y contexto.

---

## 3. Evaluación de Respuestas y Feedback Adaptativo

### 3.1 Proceso de Evaluación
1. **Embeddings:** Cada respuesta del usuario se embebe y se compara con la respuesta esperada (y con buenas respuestas previas).
2. **Similitud semántica:** Si la similitud de coseno es > 0.75 (técnico) o > 0.65 (soft), se considera correcta.
3. **Cobertura de conceptos:** Se extraen palabras clave de la respuesta esperada y se mide la cobertura en la respuesta del usuario (>60% = correcta).
4. **Intentos y pistas:** Cada intento adicional y uso de pista resta puntos.
5. **Feedback emocional:** Se analiza la emoción predominante en la respuesta (alegría, tristeza, frustración, etc.) usando un modelo transformers (`j-hartmann/emotion-english-distilroberta-base`).
6. **Feedback adaptativo:** El feedback y la motivación se ajustan dinámicamente según la emoción detectada, patrones de error y progreso.
7. **Sugerencias de recursos:** Si el usuario falla varias veces en un tema, se recomiendan recursos externos personalizados.

### 3.2 Ejemplo de Evaluación
- Respuesta: "Es para crear contenedores."
- Respuesta esperada: "Permite crear, desplegar y gestionar contenedores de aplicaciones."
- Similitud coseno: 0.81 → Correcta.
- Cobertura de conceptos: 2/3 → Correcta.
- Feedback: "¡Respuesta correcta! 🚀 Suma un logro más a tu carrera. Puntaje: 8 (intentos: 2, penalización por pistas: 2)"

---

## 4. Ajuste Dinámico de Dificultad

- El sistema ajusta la dificultad de las preguntas según la racha de aciertos/errores del usuario.
- Si responde varias preguntas seguidas correctamente, sube la dificultad (más avanzadas, menos pistas).
- Si falla repetidamente, baja la dificultad y ofrece más ayuda.
- El ajuste se realiza tanto por reglas como por ML (modulo `ml_dynamic_difficulty.py`).

---

## 5. Feedback Emocional y Motivacional

- Se detectan emociones negativas repetidas y se sugiere pausa o mensajes motivacionales.
- El feedback se adapta a la emoción detectada: si hay frustración, se motiva; si hay alegría, se refuerza el logro.
- El sistema aprende nuevas frases motivacionales y emojis usando LLMs y feedback de usuarios.

---

## 6. Aprendizaje Activo y Generación Automática de Preguntas

- Las buenas respuestas del usuario se almacenan y refrescan los embeddings para mejorar la selección y feedback futuro.
- El sistema utiliza LLMs para crear nuevas preguntas técnicas relevantes tras cada buena respuesta.
- Las nuevas preguntas se integran automáticamente al dataset y a los embeddings, permitiendo que el banco de preguntas crezca y se adapte.

---

## 7. Métricas, Analítica y Panel de Administración

- Se calculan aciertos, errores y porcentajes de éxito por tipo de pregunta y por tema (clustering).
- Se identifican habilidades destacadas y a mejorar según las respuestas y el clustering de errores.
- Se mide el tiempo de respuesta promedio y se reconoce la agilidad o se sugieren técnicas de estudio si es necesario.
- El resumen final es generado por LLM, incluye estadísticas, fortalezas, puntos de mejora, advertencias, clustering de temas, recursos recomendados y reconocimiento por agilidad.
- El panel admin permite analizar desempeño, errores frecuentes, progreso y exportar resultados.

---

## 8. Integración de Audio: STT y TTS

- El frontend permite responder por texto o voz (STT) y escuchar las respuestas del agente (TTS).
- El backend expone endpoints `/stt` y `/tts` para integración de audio.
- El temporizador se gestiona en frontend y backend para medir el tiempo total en modo examen.

---

## 9. Seguridad y Privacidad

- Todos los datos y métricas se almacenan de forma segura y anónima.
- El sistema cumple con buenas prácticas de seguridad, validación y protección de datos.

---

## 10. Extensibilidad y Futuro

- El sistema es modular y permite agregar nuevos modelos, algoritmos ML, visualizaciones y recursos fácilmente.
- El feedback y la selección de preguntas se pueden mejorar con modelos más avanzados o datos adicionales.

---

**Última actualización: agosto 2025**
