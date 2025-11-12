using Microsoft.AspNetCore.Components;
using Microsoft.EntityFrameworkCore;
using Microsoft.JSInterop;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.MVVM.ViewModels;
using Ready4Hire.Services;

namespace Ready4Hire.MVVM.Views
{
    public partial class ChatPage : ComponentBase, IDisposable
    {
        // Servicio del agente
        [Inject]
        public InterviewApiService InterviewApi { get; set; } = null!;
        
        // Factory de base de datos (para evitar concurrencia)
        [Inject]
        private IDbContextFactory<AppDbContext> DbFactory { get; set; } = null!;

        [Inject]
        private AuthService AuthService { get; set; } = null!;

        [Inject]
        private SecurityService SecurityService { get; set; } = null!;

        [Inject]
        private NavigationManager Navigation { get; set; } = null!;


        // Id chat
        [Parameter]
        public int chatId { get; set; }

        private ChatViewModel? vm;
        private AppDbContext? currentDb;
        private List<Message> Messages = new();

        private string UserInput { get; set; } = "";
        private string userId = $"user-{Guid.NewGuid().ToString().Substring(0, 8)}";
        private string interviewId = "";  // ID de la entrevista actual
        private string interviewType = "technical";
        private string mode = "practice";
        private string currentQuestion = "";
        private string currentPhase = "config";  // config | context | questions | completed
        private int questionCount = 0;
        private int contextQuestionsAnswered = 0;
        private bool started = false;
        private ElementReference chatBodyRef;
        private bool isExamMode = false;
        private System.Timers.Timer? examTimer;
        private int elapsedSeconds = 0;
        private string timerDisplay = "00:00";
        private string? errorMessage = null;

        /// Indica si se muestra el modal de configuración.
        public bool ShowConfig { get; set; } = false;

        // Referencias a elementos del DOM
    private ElementReference chatMessagesContainer;
    private ElementReference inputTextArea;
    
    // Usuario actual para el saludo personalizado
    private User? currentUser;

        // Configuración de la entrevista
        private string SelectedRole { get; set; } = "";
        private string SelectedInterviewType { get; set; } = "technical";
        private string SelectedDifficulty { get; set; } = "mid";

        // Propiedades calculadas
        private bool IsConfigured => !string.IsNullOrEmpty(SelectedRole);
        private bool CanSendMessage => !string.IsNullOrWhiteSpace(UserInput) && started;

        private string FormatTime(int seconds)
        {
            var minutes = seconds / 60;
            var secs = seconds % 60;
            return $"{minutes:00}:{secs:00}";
        }

        /// Modo de entrevista seleccionado ("practice" o "exam").
        public string SelectedMode { get; set; } = "practice";

        /// Indica si la configuración fue guardada y es válida.
        public bool IsConfiguredLegacy { get; set; } = false;

        // Métodos necesarios para el .razor
        private string FormatMessageMethod(string message)
        {
            // Conversión básica de markdown/texto
            return message.Replace("\n", "<br/>");
        }

        private async Task HandleKeyPressMethod(Microsoft.AspNetCore.Components.Web.KeyboardEventArgs e)
        {
            if (e.Key == "Enter" && !e.ShiftKey && CanSendMessage)
            {
                await SendMessage();
            }
        }

        private bool isAuthenticated = false;

        // Variables para STT/TTS
        private bool isRecording = false;
        private bool isPlayingTTS = false;
        private DotNetObjectReference<ChatPage>? dotNetRef;
        private IJSObjectReference? mediaRecorder;

        protected override async Task OnInitializedAsync()
        {
            // Crear una nueva instancia de DbContext para este componente
            currentDb = await DbFactory.CreateDbContextAsync();
            vm = new ChatViewModel(currentDb, chatId);
            await vm.LoadDataAsync(chatId);

            // Cargar el estado existente de la entrevista si existe
            await LoadExistingInterviewState();
            
            // Cargar datos del usuario actual para el saludo
            await LoadCurrentUser();

            // Inicializar la referencia para JavaScript
            dotNetRef = DotNetObjectReference.Create(this);
        }

        /// <summary>
        /// Carga el estado existente de una entrevista si ya hay datos guardados
        /// </summary>
        private async Task LoadExistingInterviewState()
        {
            if (chatId != 0 && vm?.Messages != null && vm.Messages.Any())
            {
                Messages = vm.Messages;
                
                // Si ya hay mensajes, significa que la entrevista ya comenzó
                started = true;
                currentPhase = "context"; // Asumir que está en progreso
                
                // Contar preguntas de contexto respondidas aproximadamente
                contextQuestionsAnswered = Messages.Count(m => m.IsUser) / 2; // Estimación
                questionCount = Messages.Count(m => m.IsUser);
                
                StateHasChanged();
            }
            else if (chatId != 0)
            {
                // Si existe el chat pero no tiene mensajes, mostrar configuración
                started = false;
            }
        }

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                // SEGURIDAD: Verificar autenticación después del primer render (cuando JSInterop está disponible)
                if (!await AuthService.IsAuthenticatedAsync())
                {
                    Navigation.NavigateTo("/", true);
                    return;
                }

                // Validar sesión activa
                if (!await AuthService.ValidateSessionAsync())
                {
                    await AuthService.LogoutAsync();
                    Navigation.NavigateTo("/", true);
                    return;
                }

                isAuthenticated = true;
                StateHasChanged();
            }

            await base.OnAfterRenderAsync(firstRender);
            await ScrollToBottomAsync();
        }


        /// Muestra el modal de configuración de entrevista.
        private void ShowSetup()
        {
            ShowConfig = true;
        }

        /// Oculta el modal de configuración de entrevista.
        private void HideSetup()
        {
            ShowConfig = false;
        }

        /// Guarda la configuración seleccionada y habilita el inicio de la entrevista.
        private void SaveConfig()
        {
            interviewType = SelectedInterviewType;
            mode = SelectedMode;
                IsConfiguredLegacy = true;
            ShowConfig = false;
            isExamMode = (mode == "exam");
            StateHasChanged();
        }

        /// <summary>
        /// [V2] Inicia la entrevista con fase de contexto.
        /// 1. Llama a POST /api/v2/interviews
        /// 2. Obtiene interview_id
        /// 3. Muestra primera pregunta de contexto
        /// 4. Actualiza currentPhase = "context"
        /// </summary>
        private async Task StartInterview()
        {
            if (!IsConfigured) return;

            Messages.Clear();
            questionCount = 0;
            contextQuestionsAnswered = 0;
            started = true;
            errorMessage = null;
            
            if (isExamMode)
            {
                StartExamTimer();
            }
            else
            {
                StopExamTimer();
            }

            try
            {
                // Llamar al nuevo endpoint V2
                var startResponse = await InterviewApi.StartInterviewV2Async(
                    userId, 
                    SelectedRole, 
                    SelectedInterviewType,
                    SelectedDifficulty
                );

                interviewId = startResponse.GetProperty("interview_id").GetString();
                currentPhase = startResponse.GetProperty("status").GetString(); // "context"

                // Mostrar mensaje de bienvenida
                Messages.Add(new Message 
                { 
                    Text = "🎯 ¡Bienvenido a tu entrevista personalizada! Primero, me gustaría conocerte mejor. Responde 5 preguntas de contexto para personalizar tu experiencia.", 
                    IsUser = false 
                });

                // Mostrar primera pregunta de contexto
                var firstQuestion = startResponse.GetProperty("first_question")
                    .GetProperty("text").GetString();
                Messages.Add(new Message { Text = firstQuestion, IsUser = false });

                StateHasChanged();
            }
            catch (TimeoutException ex)
            {
                errorMessage = "⏱️ Tiempo de espera agotado. Verifica que el servidor backend esté ejecutándose.";
                StartOfflineMode();
            }
            catch (HttpRequestException ex)
            {
                errorMessage = $"🌐 Error de conexión: No se puede conectar al servidor. {ex.Message}";
                StartOfflineMode();
            }
            catch (InvalidOperationException ex)
            {
                if (ex.Message.Contains("métricas de Prometheus"))
                {
                    errorMessage = "🔧 El backend está ejecutándose en el puerto incorrecto o hay un problema de configuración de rutas. Activando modo offline.";
                }
                else
                {
                    errorMessage = $"⚠️ {ex.Message}";
                }
                StartOfflineMode();
            }
            catch (Exception ex)
            {
                errorMessage = $"❌ Error inesperado: {ex.Message}";
                StartOfflineMode();
            }

            await ScrollToBottomAsync();
        }

        /// <summary>
        /// [V2] Envía la respuesta del usuario a la API y procesa la respuesta del agente.
        /// Maneja:
        /// 1. Fase de contexto (5 preguntas)
        /// 2. Transición a preguntas técnicas/soft skills
        /// 3. Feedback y motivación
        /// 4. Intentos múltiples (máximo 3)
        /// 5. Progreso de la entrevista
        /// </summary>
        private async Task SendMessage()
        {
            if (string.IsNullOrWhiteSpace(UserInput) || string.IsNullOrEmpty(interviewId))
                return;

            var answer = SanitizeInput(UserInput);
            Messages.Add(new Message { Text = answer, IsUser = true });
            UserInput = "";
            errorMessage = null;
            StateHasChanged();
            await ScrollToBottomAsync();

            try
            {
                // Llamar al endpoint V2 ProcessAnswer
                var response = await InterviewApi.ProcessAnswerV2Async(interviewId, answer, elapsedSeconds);

                // Actualizar fase actual
                if (response.TryGetProperty("phase", out var phaseProperty))
                {
                    currentPhase = phaseProperty.GetString();
                }

                // Mostrar feedback si existe
                if (response.TryGetProperty("feedback", out var feedback))
                {
                    var feedbackText = feedback.GetString();
                    if (!string.IsNullOrEmpty(feedbackText))
                    {
                        Messages.Add(new Message { Text = feedbackText, IsUser = false });
                    }
                }

                // Mostrar motivación si existe
                if (response.TryGetProperty("motivation", out var motivation))
                {
                    var motivationText = motivation.GetString();
                    if (!string.IsNullOrEmpty(motivationText))
                    {
                        Messages.Add(new Message { Text = "💪 " + motivationText, IsUser = false });
                    }
                }

                // Actualizar progreso
                if (response.TryGetProperty("progress", out var progress))
                {
                    contextQuestionsAnswered = progress.GetProperty("context_completed").GetInt32();
                    questionCount = progress.GetProperty("questions_completed").GetInt32();
                }

                // Mostrar siguiente pregunta si existe (puede venir como "question" o "next_question")
                var questionProperty = response.TryGetProperty("next_question", out var nextQuestion) 
                    ? nextQuestion 
                    : (response.TryGetProperty("question", out var question) ? question : default);
                    
                if (questionProperty.ValueKind != System.Text.Json.JsonValueKind.Undefined && 
                    questionProperty.ValueKind != System.Text.Json.JsonValueKind.Null)
                {
                    var questionText = questionProperty.GetProperty("text").GetString();
                    if (!string.IsNullOrEmpty(questionText))
                    {
                        Messages.Add(new Message { Text = questionText, IsUser = false });
                    }

                    // Mostrar intentos restantes si está en retry
                    if (questionProperty.TryGetProperty("retry", out var retry) && retry.GetBoolean())
                    {
                        if (response.TryGetProperty("attempts_left", out var attemptsProperty))
                        {
                            var attemptsLeft = attemptsProperty.GetInt32();
                            Messages.Add(new Message 
                            { 
                                Text = $"ℹ️ Te quedan {attemptsLeft} intentos para esta pregunta.", 
                                IsUser = false 
                            });
                        }
                    }
                }

                // Si completó la entrevista
                if (currentPhase == "completed")
                {
                    Messages.Add(new Message 
                    { 
                        Text = "� ¡Felicidades! Has completado la entrevista. Gracias por tu participación.", 
                        IsUser = false 
                    });
                }

                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error: {ex.Message}";
                Messages.Add(new Message 
                { 
                    Text = $"❌ Error al procesar tu respuesta. Por favor, intenta de nuevo.", 
                    IsUser = false 
                });
            }

            StateHasChanged();
            await ScrollToBottomAsync();
        }

        /// <summary>
        /// Finaliza la entrevista y muestra el resumen devuelto por la API.
        /// </summary>
        /// <summary>
        /// Inicia el modo offline cuando el backend no está disponible
        /// </summary>
        private void StartOfflineMode()
        {
            Messages.Add(new Message 
            { 
                Text = "🔄 Modo offline activado. Simulando entrevista para demostración.", 
                IsUser = false 
            });
            
            Messages.Add(new Message 
            { 
                Text = "📝 ¿Puedes contarme sobre tu experiencia profesional y qué te motiva en tu carrera?", 
                IsUser = false 
            });
            
            interviewId = "offline_" + Guid.NewGuid().ToString("N")[..8];
            currentPhase = "context";
            StateHasChanged();
        }

        /// <summary>
        /// Limpia el mensaje de error y permite continuar
        /// </summary>
        private void ClearError()
        {
            errorMessage = null;
            StateHasChanged();
        }

        private void ToggleExamMode()
        {
            isExamMode = !isExamMode;
            StateHasChanged();
        }

        /// <summary>
        /// Carga los datos del usuario actual
        /// </summary>
        private async Task LoadCurrentUser()
        {
            try
            {
                var email = await AuthService.GetCurrentUserEmailAsync();
                if (!string.IsNullOrEmpty(email) && currentDb != null)
                {
                    currentUser = await currentDb.Users
                        .FirstOrDefaultAsync(u => u.Email == email);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading current user: {ex.Message}");
            }
        }

        /// <summary>
        /// Obtiene el nombre del usuario para mostrar en el saludo
        /// </summary>
        private string GetUserName()
        {
            return currentUser?.Name ?? "Usuario";
        }

        /// <summary>
        /// Guarda un mensaje en la base de datos para persistir el historial
        /// </summary>
        private async Task SaveMessageToDatabase(string messageText, bool isUser)
        {
            try
            {
                if (currentDb != null && chatId != 0)
                {
                    var message = new Message
                    {
                        ChatId = chatId,
                        Text = messageText,
                        IsUser = isUser,
                        Timestamp = DateTime.UtcNow
                    };

                    currentDb.Messages.Add(message);
                    await currentDb.SaveChangesAsync();

                    // Actualizar también la lista en memoria
                    Messages.Add(new Message 
                    { 
                        Text = messageText, 
                        IsUser = isUser 
                    });
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving message to database: {ex.Message}");
                // Si falla el guardado en DB, al menos mantener en memoria
                Messages.Add(new Message 
                { 
                    Text = messageText, 
                    IsUser = isUser 
                });
            }
        }

        private async Task EndInterview()
        {
            StopExamTimer();
            try
            {
                if (interviewId?.StartsWith("offline_") == true)
                {
                    // Modo offline
                    Messages.Add(new Message 
                    { 
                        Text = "🎯 Resumen de entrevista (Modo Demo):\n\n" +
                               "✅ Participación activa demostrada\n" +
                               "💡 Respuestas reflexivas y coherentes\n" +
                               "🚀 Potencial para crecimiento profesional\n\n" +
                               "🎮 ¡Obtuviste 150 puntos de experiencia!\n" +
                               "🏆 Insignia desbloqueada: 'Primer Paso'", 
                        IsUser = false 
                    });
                }
                else
                {
                    var result = await InterviewApi.EndInterviewAsync(userId);
                    if (result.TryGetProperty("summary", out var summary))
                    {
                        var summaryText = summary.GetString() ?? "";
                        // Gamificación: si contiene el bloque especial, mostrarlo resaltado
                        if (summaryText.Contains("🎮 Sistema de Gamificación Avanzada 🎮"))
                        {
                            Messages.Add(new Message { Text = summaryText, IsUser = false });
                        }
                        else
                        {
                            Messages.Add(new Message { Text = summaryText, IsUser = false });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al finalizar la entrevista: {ex.Message}";
            }
            started = false;
            StateHasChanged();
            await ScrollToBottomAsync();
        }
        /// <summary>
        /// Procesa respuestas en modo offline para demostración
        /// </summary>
        private void ProcessOfflineAnswer(string answer)
        {
            contextQuestionsAnswered++;
            questionCount++;

            // Respuestas simuladas inteligentes
            var feedbackResponses = new[]
            {
                "💡 Excelente respuesta. Me parece que tienes una perspectiva muy clara.",
                "👏 Interesante enfoque. Eso demuestra tu capacidad analítica.",
                "✨ Muy bien explicado. Tu experiencia se nota en la respuesta.",
                "🎯 Perfecto. Esa actitud es exactamente lo que buscamos.",
                "🚀 Impresionante. Tu pasión por la tecnología es evidente."
            };

            var questions = new[]
            {
                "🔧 ¿Qué herramientas o tecnologías prefieres usar en tu trabajo y por qué?",
                "📚 ¿Cómo te mantienes actualizado con las últimas tendencias en tu campo?",
                "🤝 Cuéntame sobre una vez que tuviste que trabajar en equipo para resolver un problema difícil.",
                "🎯 ¿Cuáles son tus objetivos profesionales a largo plazo?",
                "💡 ¿Puedes describir un proyecto del que te sientes especialmente orgulloso?"
            };

            // Feedback simulado
            var randomFeedback = feedbackResponses[Random.Shared.Next(feedbackResponses.Length)];
            Messages.Add(new Message { Text = randomFeedback, IsUser = false });

            // Siguiente pregunta si no hemos terminado
            if (contextQuestionsAnswered < 5)
            {
                var nextQuestion = questions[Math.Min(contextQuestionsAnswered, questions.Length - 1)];
                Messages.Add(new Message { Text = nextQuestion, IsUser = false });
            }
            else
            {
                currentPhase = "completed";
                Messages.Add(new Message 
                { 
                    Text = "🎉 ¡Excelente! Has completado todas las preguntas de contexto. " +
                           "En una entrevista real, ahora pasaríamos a las preguntas técnicas específicas. " +
                           "¿Te gustaría finalizar esta sesión de demostración?", 
                    IsUser = false 
                });
            }
        }

        // Sanitización similar a chat.js
        private string SanitizeInput(string text)
        {
            if (string.IsNullOrEmpty(text)) return "";
            var sanitized = text.Replace("<", "").Replace(">", "").Replace("\"", "").Replace("'", "").Replace("`", "");
            sanitized = System.Text.RegularExpressions.Regex.Replace(sanitized, "\\s+", " ").Trim();
            sanitized = sanitized.Normalize(System.Text.NormalizationForm.FormD);
            var sb = new System.Text.StringBuilder();
            foreach (var c in sanitized)
            {
                var uc = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(c);
                if (uc != System.Globalization.UnicodeCategory.NonSpacingMark)
                    sb.Append(c);
            }
            return sb.ToString().Normalize(System.Text.NormalizationForm.FormC);
        }
        // Temporizador para modo examen
        private void StartExamTimer()
        {
            StopExamTimer();
            elapsedSeconds = 0;
            timerDisplay = "00:00";
            examTimer = new System.Timers.Timer(1000);
            examTimer.Elapsed += (s, e) =>
            {
                elapsedSeconds++;
                var min = (elapsedSeconds / 60).ToString("D2");
                var sec = (elapsedSeconds % 60).ToString("D2");
                timerDisplay = $"{min}:{sec}";
                InvokeAsync(StateHasChanged);
            };
            examTimer.AutoReset = true;
            examTimer.Start();
        }

        private void StopExamTimer()
        {
            if (examTimer != null)
            {
                examTimer.Stop();
                examTimer.Dispose();
                examTimer = null;
            }
            timerDisplay = "00:00";
        }

        /// Hace scroll automático al final del chat usando JS interop.
        private async Task ScrollToBottomAsync()
        {
            try
            {
                // Scroll functionality - simplified for now
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error scrolling: {ex.Message}");
            }
        }

        /// Redimensiona automáticamente el textarea según su contenido
        private async Task AutoResizeTextarea()
        {
            try
            {
                // Auto-resize functionality - simplified for now
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error resizing: {ex.Message}");
            }
        }


        /// Maneja el evento de teclado para enviar con Enter
        private async Task HandleKeyDown(Microsoft.AspNetCore.Components.Web.KeyboardEventArgs e)
        {
            if (e.Key == "Enter" && !e.ShiftKey && !string.IsNullOrWhiteSpace(UserInput))
            {
                await SendMessage();
            }
        }

        /// Obtiene las iniciales del usuario para el avatar
        private string GetUserInitials()
        {
            if (vm?.User == null)
                return "U";
            
            var firstName = vm.User.Name?.Trim() ?? "";
            var lastName = vm.User.LastName?.Trim() ?? "";
            
            var initials = "";
            if (!string.IsNullOrEmpty(firstName))
                initials += firstName[0];
            if (!string.IsNullOrEmpty(lastName))
                initials += lastName[0];
            
            return string.IsNullOrEmpty(initials) ? "U" : initials.ToUpper();
        }

        /// Obtiene el nombre del usuario
        private string GetUserFirstName()
        {
            return vm?.User?.Name ?? "Usuario";
        }

        #region STT/TTS Methods

        /// <summary>
        /// Alterna entre iniciar y detener la grabación de audio (STT)
        /// </summary>
        private async Task ToggleRecording()
        {
            try
            {
                if (!isRecording)
                {
                    await StartRecording();
                }
                else
                {
                    await StopRecording();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error en ToggleRecording: {ex.Message}");
                // Mostrar mensaje de error al usuario
                errorMessage = "Error al acceder al micrófono. Verifica los permisos.";
                StateHasChanged();
            }
        }

        /// <summary>
        /// Inicia la grabación de audio
        /// </summary>
        private async Task StartRecording()
        {
            try
            {
                // Verificar soporte del navegador
                var isSupported = await JSRuntime.InvokeAsync<bool>("isMediaRecorderSupported");
                if (!isSupported)
                {
                    errorMessage = "Tu navegador no soporta grabación de audio.";
                    StateHasChanged();
                    return;
                }

                // Solicitar permisos
                var hasPermission = await JSRuntime.InvokeAsync<bool>("requestMicrophonePermission");
                if (!hasPermission)
                {
                    errorMessage = "Se requieren permisos de micrófono para la grabación.";
                    StateHasChanged();
                    return;
                }

                // Inicializar MediaRecorder
                mediaRecorder = await JSRuntime.InvokeAsync<IJSObjectReference>("initializeMediaRecorder");
                
                // Iniciar grabación
                await JSRuntime.InvokeVoidAsync("startRecording", mediaRecorder);
                
                isRecording = true;
                errorMessage = null;
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al iniciar grabación: {ex.Message}";
                isRecording = false;
                StateHasChanged();
            }
        }

        /// <summary>
        /// Detiene la grabación y procesa el audio con STT
        /// </summary>
        private async Task StopRecording()
        {
            try
            {
                if (mediaRecorder == null) return;

                // Detener grabación y obtener el blob
                var audioBlob = await JSRuntime.InvokeAsync<IJSObjectReference>("stopRecording", mediaRecorder);
                
                // Convertir a bytes
                var audioBytes = await JSRuntime.InvokeAsync<byte[]>("blobToBytes", audioBlob);
                
                isRecording = false;
                StateHasChanged();

                // Procesar con STT
                await ProcessSpeechToText(audioBytes);
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al procesar audio: {ex.Message}";
                isRecording = false;
                StateHasChanged();
            }
        }

        /// <summary>
        /// Procesa el audio con el servicio STT
        /// </summary>
        private async Task ProcessSpeechToText(byte[] audioBytes)
        {
            try
            {
                // Llamar al servicio STT del backend
                var result = await InterviewApi.SpeechToTextAsync(audioBytes, "es");
                
                // Extraer el texto transcrito
                if (result.TryGetProperty("text", out var textElement))
                {
                    var transcribedText = textElement.GetString();
                    if (!string.IsNullOrWhiteSpace(transcribedText))
                    {
                        UserInput = transcribedText;
                        StateHasChanged();
                    }
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Error en transcripción: {ex.Message}";
                StateHasChanged();
            }
        }

        /// <summary>
        /// Alterna la reproducción de audio del último mensaje (TTS)
        /// </summary>
        private async Task ToggleTTS()
        {
            try
            {
                if (isPlayingTTS)
                {
                    await StopTTS();
                }
                else
                {
                    await StartTTS();
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Error en TTS: {ex.Message}";
                isPlayingTTS = false;
                StateHasChanged();
            }
        }

        /// <summary>
        /// Inicia la reproducción TTS del último mensaje del asistente
        /// </summary>
        private async Task StartTTS()
        {
            try
            {
                // Obtener el último mensaje del asistente
                var lastAssistantMessage = Messages.LastOrDefault(m => !m.IsUser);
                if (lastAssistantMessage == null)
                {
                    errorMessage = "No hay mensajes para reproducir.";
                    StateHasChanged();
                    return;
                }

                // Llamar al servicio TTS
                var audioBytes = await InterviewApi.TextToSpeechAsync(lastAssistantMessage.Text, "es");
                
                // Crear elemento audio y reproducir
                var audioElement = await JSRuntime.InvokeAsync<IJSObjectReference>("createAudioFromBytes", audioBytes);
                
                // Configurar callback para cuando termine
                await JSRuntime.InvokeVoidAsync("setupAudioEndCallback", audioElement, dotNetRef);
                
                // Reproducir
                await JSRuntime.InvokeVoidAsync("playAudio", audioElement);
                
                isPlayingTTS = true;
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al reproducir audio: {ex.Message}";
                isPlayingTTS = false;
                StateHasChanged();
            }
        }

        /// <summary>
        /// Detiene la reproducción TTS
        /// </summary>
        private async Task StopTTS()
        {
            try
            {
                // Esta funcionalidad se puede implementar si se necesita detener manualmente
                isPlayingTTS = false;
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al detener TTS: {ex.Message}");
            }
        }

        /// <summary>
        /// Callback de JavaScript cuando termina la reproducción de audio
        /// </summary>
        [JSInvokable]
        public void OnAudioEnded()
        {
            isPlayingTTS = false;
            InvokeAsync(StateHasChanged);
        }

        #endregion

        /// <summary>
        /// Libera los recursos utilizados
        /// </summary>
        public void Dispose()
        {
            try
            {
                dotNetRef?.Dispose();
                mediaRecorder?.DisposeAsync();
                currentDb?.Dispose();
                examTimer?.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error disposing resources: {ex.Message}");
            }
        }
    }
}
