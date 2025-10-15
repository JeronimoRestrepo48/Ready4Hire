using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.MVVM.Views
{
    public partial class ChatPage : ComponentBase
    {

        [Inject]
        public InterviewApiService InterviewApi { get; set; }


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

        /// Tipo de entrevista seleccionado ("technical" o "soft_skills").
        public string SelectedInterviewType { get; set; } = "technical";

        /// Rol seleccionado (ej: "Backend Developer", "Frontend Developer").
        public string SelectedRole { get; set; } = "Backend Developer";

        /// Dificultad seleccionada ("junior", "mid", "senior").
        public string SelectedDifficulty { get; set; } = "junior";

        /// Modo de entrevista seleccionado ("practice" o "exam").
        public string SelectedMode { get; set; } = "practice";

        /// Indica si la configuración fue guardada y es válida.
        public bool IsConfigured { get; set; } = false;

        
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
            IsConfigured = true;
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
            catch (Exception ex)
            {
                errorMessage = $"Error al iniciar la entrevista: {ex.Message}";
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

                // Mostrar siguiente pregunta si existe
                if (response.TryGetProperty("question", out var question))
                {
                    var questionText = question.GetProperty("text").GetString();
                    Messages.Add(new Message { Text = questionText, IsUser = false });

                    // Mostrar intentos restantes si está en retry
                    if (question.TryGetProperty("retry", out var retry) && retry.GetBoolean())
                    {
                        var attemptsLeft = response.GetProperty("attempts_left").GetInt32();
                        Messages.Add(new Message 
                        { 
                            Text = $"ℹ️ Te quedan {attemptsLeft} intentos para esta pregunta.", 
                            IsUser = false 
                        });
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
        private async Task EndInterview()
        {
            StopExamTimer();
            try
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
            catch (Exception ex)
            {
                errorMessage = $"Error al finalizar la entrevista: {ex.Message}";
            }
            started = false;
            StateHasChanged();
            await ScrollToBottomAsync();
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

        /// <summary>
        /// Hace scroll automático al final del chat usando JS interop.
        /// </summary>
        private async Task ScrollToBottomAsync()
        {
            await JS.InvokeVoidAsync("scrollToBottom");
        }

        /// <summary>
        /// Hace scroll automático después de renderizar el componente.
        /// </summary>
        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            await ScrollToBottomAsync();
        }
    }
}
