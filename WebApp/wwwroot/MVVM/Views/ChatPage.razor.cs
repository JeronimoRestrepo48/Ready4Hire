using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;

namespace Ready4Hire.MVVM.Views
{
    public partial class ChatPage : ComponentBase
    {
        /// <summary>
        /// Representa un mensaje en el chat (usuario o agente).
        /// </summary>
        public class ChatMessage
        {
            /// <summary>
            /// Texto del mensaje.
            /// </summary>
            public string Text { get; set; }
            /// <summary>
            /// Indica si el mensaje es del usuario (true) o del agente (false).
            /// </summary>
            public bool IsUser { get; set; }
        }

        [Inject]
        public Ready4Hire.MVVM.Models.InterviewApiService InterviewApi { get; set; }


    private List<ChatMessage> Messages = new();
    private string UserInput { get; set; } = "";
    private string userId = $"user-{Guid.NewGuid().ToString().Substring(0, 8)}";
    private string interviewType = "technical";
    private string mode = "practice";
    private string currentQuestion = "";
    private int questionCount = 0;
    private const int MAX_QUESTIONS = 10;
    private bool started = false;
    private ElementReference chatBodyRef;
    private bool isExamMode = false;
    private System.Timers.Timer? examTimer;
    private int elapsedSeconds = 0;
    private string timerDisplay = "00:00";
    private string? errorMessage = null;

    /// <summary>
    /// Indica si se muestra el modal de configuración.
    /// </summary>
    public bool ShowConfig { get; set; } = false;
    /// <summary>
    /// Tipo de entrevista seleccionado ("technical" o "soft").
    /// </summary>
    public string SelectedInterviewType { get; set; } = "technical";
    /// <summary>
    /// Modo de entrevista seleccionado ("practice" o "exam").
    /// </summary>
    public string SelectedMode { get; set; } = "practice";
    /// <summary>
    /// Indica si la configuración fue guardada y es válida.
    /// </summary>
    public bool IsConfigured { get; set; } = false;

        /// <summary>
        /// Muestra el modal de configuración de entrevista.
        /// </summary>
        private void ShowSetup()
        {
            ShowConfig = true;
        }

        /// <summary>
        /// Oculta el modal de configuración de entrevista.
        /// </summary>
        private void HideSetup()
        {
            ShowConfig = false;
        }

        /// <summary>
        /// Guarda la configuración seleccionada y habilita el inicio de la entrevista.
        /// </summary>
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
        /// Inicia la entrevista llamando a la API Python con los parámetros configurados.
        /// </summary>
        private async Task StartInterview()
        {
            if (!IsConfigured) return;
            Messages.Clear();
            questionCount = 0;
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
                var result = await InterviewApi.StartInterviewAsync(userId, string.Empty, interviewType, mode);
                if (result.TryGetProperty("question", out var question))
                {
                    var qStr = question.GetString();
                    if (!string.IsNullOrEmpty(qStr))
                    {
                        currentQuestion = qStr;
                        Messages.Add(new ChatMessage { Text = currentQuestion, IsUser = false });
                    }
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al iniciar la entrevista: {ex.Message}";
            }
            await ScrollToBottomAsync();
        }

        /// <summary>
        /// Envía la respuesta del usuario a la API y procesa la respuesta del agente.
        /// </summary>
        private async Task SendMessage()
        {
            if (!string.IsNullOrWhiteSpace(UserInput) && started)
            {
                var sanitized = SanitizeInput(UserInput);
                Messages.Add(new ChatMessage { Text = sanitized, IsUser = true });
                var answer = sanitized;
                UserInput = "";
                errorMessage = null;
                StateHasChanged();
                await ScrollToBottomAsync();

                try
                {
                    var result = await InterviewApi.AnswerAsync(userId, answer);
                    // Manejo de retry
                    bool retry = false;
                    if (result.TryGetProperty("retry", out var retryProp) && retryProp.ValueKind == System.Text.Json.JsonValueKind.True)
                    {
                        retry = true;
                    }
                    if (result.TryGetProperty("next", out var nextQ))
                    {
                        var nextStr = nextQ.GetString();
                        if (!string.IsNullOrEmpty(nextStr))
                        {
                            currentQuestion = nextStr;
                            Messages.Add(new ChatMessage { Text = currentQuestion, IsUser = false });
                        }
                        if (!retry)
                        {
                            questionCount++;
                        }
                    }
                    else if (result.TryGetProperty("feedback", out var feedback))
                    {
                        var feedbackText = feedback.GetString() ?? "";
                        // Gamificación: si contiene el bloque especial, mostrarlo resaltado
                        if (feedbackText.Contains("🎮 Sistema de Gamificación Avanzada 🎮"))
                        {
                            Messages.Add(new ChatMessage { Text = feedbackText, IsUser = false });
                        }
                        else
                        {
                            Messages.Add(new ChatMessage { Text = feedbackText, IsUser = false });
                        }
                        if (!retry)
                        {
                            questionCount++;
                        }
                    }
                    // Solo avanzar si retry es false
                    if (!retry && questionCount >= MAX_QUESTIONS)
                    {
                        await EndInterview();
                    }
                }
                catch (Exception ex)
                {
                    errorMessage = $"Error al enviar respuesta: {ex.Message}";
                }
                await ScrollToBottomAsync();
            }
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
                        Messages.Add(new ChatMessage { Text = summaryText, IsUser = false });
                    }
                    else
                    {
                        Messages.Add(new ChatMessage { Text = summaryText, IsUser = false });
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
