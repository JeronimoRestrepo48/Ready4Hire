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
            var result = await InterviewApi.StartInterviewAsync(userId, null, interviewType, mode);
            if (result.TryGetProperty("question", out var question))
            {
                currentQuestion = question.GetString();
                Messages.Add(new ChatMessage { Text = currentQuestion, IsUser = false });
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
                Messages.Add(new ChatMessage { Text = UserInput.Trim(), IsUser = true });
                var answer = UserInput.Trim();
                UserInput = "";
                StateHasChanged();
                await ScrollToBottomAsync();

                var result = await InterviewApi.AnswerAsync(userId, answer);
                if (result.TryGetProperty("next", out var nextQ))
                {
                    currentQuestion = nextQ.GetString();
                    Messages.Add(new ChatMessage { Text = currentQuestion, IsUser = false });
                }
                else if (result.TryGetProperty("feedback", out var feedback))
                {
                    Messages.Add(new ChatMessage { Text = feedback.GetString(), IsUser = false });
                }
                questionCount++;
                if (questionCount >= MAX_QUESTIONS)
                {
                    await EndInterview();
                }
                await ScrollToBottomAsync();
            }
        }

        /// <summary>
        /// Finaliza la entrevista y muestra el resumen devuelto por la API.
        /// </summary>
        private async Task EndInterview()
        {
            var result = await InterviewApi.EndInterviewAsync(userId);
            if (result.TryGetProperty("summary", out var summary))
            {
                Messages.Add(new ChatMessage { Text = summary.GetString(), IsUser = false });
            }
            started = false;
            StateHasChanged();
            await ScrollToBottomAsync();
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
