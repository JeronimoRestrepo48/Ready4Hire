using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;

namespace Ready4Hire.MVVM.Views
{
    public partial class ChatPage : ComponentBase
    {
        public class ChatMessage
        {
            public string Text { get; set; }
            public bool IsUser { get; set; }
        }

        private List<ChatMessage> Messages = new()
        {
        new ChatMessage { Text = "¡Hola! Soy tu Entrenador AI de Entrevistas. Estoy aquí para ayudarte a prepararte para tu entrevista. ¿Con qué tipo de entrevista te gustaría empezar o qué temas te gustaría cubrir?", IsUser = false },
        new ChatMessage { Text = "Quiero practicar preguntas comunes de entrevistas de desarrollo de software.", IsUser = true },
        new ChatMessage { Text = "Perfecto. Empecemos con \"Háblame de ti\". ¿Cómo responderías a esta pregunta? Tómate tu tiempo para formular una respuesta concisa y relevante.", IsUser = false }
        };

        private string UserInput { get; set; } = "";

        private ElementReference chatBodyRef;
        private async Task SendMessage()
        {
            if (!string.IsNullOrWhiteSpace(UserInput))
            {
                Messages.Add(new ChatMessage { Text = UserInput.Trim(), IsUser = true });
                UserInput = "";
                await ScrollToBottomAsync();
                StateHasChanged();

                // llamar a tu servicio de IA para obtener la respuesta del agente
                //($"post/start_interview{id},{type},{mode}");
            }
        }

        private async Task ScrollToBottomAsync()
        {
            await JS.InvokeVoidAsync("scrollToBottom");
        }

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            await ScrollToBottomAsync();
        }
    }
}
