using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text.Json;
using System.Text;
using Microsoft.Extensions.Configuration;

namespace Ready4Hire.MVVM.Models
{
    /// <summary>
    /// Servicio para consumir la API Python de entrevistas Ready4Hire.
    /// Proporciona métodos para iniciar, responder y finalizar entrevistas, así como obtener roles, niveles, preguntas y servicios de audio.
    /// </summary>
    public class InterviewApiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public InterviewApiService(HttpClient httpClient, IConfiguration configuration)
        {
            _httpClient = httpClient;
            _baseUrl = configuration["Ready4HireApi:BaseUrl"] ?? "http://localhost:8001";
        }

        // 1. Start Interview
    /// <summary>
    /// Inicia una nueva entrevista para un usuario.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    /// <param name="role">Rol (puede ser null)</param>
    /// <param name="type">Tipo de entrevista ("technical" o "soft")</param>
    /// <param name="mode">Modo ("practice" o "exam")</param>
    public async Task<JsonElement> StartInterviewAsync(string userId, string role, string type, string mode = "practice")
    {
            var payload = new { user_id = userId, role, type, mode };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/start_interview", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 2. Next Question
    /// <summary>
    /// Solicita la siguiente pregunta de la entrevista para un usuario.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    public async Task<JsonElement> NextQuestionAsync(string userId)
    {
            var payload = new { user_id = userId };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/next_question", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 3. Answer
    /// <summary>
    /// Envía una respuesta del usuario y obtiene feedback o la siguiente pregunta.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    /// <param name="answer">Respuesta del usuario</param>
    public async Task<JsonElement> AnswerAsync(string userId, string answer)
    {
            var payload = new { user_id = userId, answer = answer };
            var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync($"{_baseUrl}/answer", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 4. End Interview
    /// <summary>
    /// Finaliza la entrevista para un usuario y obtiene el resumen.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    public async Task<JsonElement> EndInterviewAsync(string userId)
    {
            var content = new StringContent($"\"{userId}\"", Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync($"{_baseUrl}/end_interview", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 5. Get Roles
    /// <summary>
    /// Obtiene la lista de roles/cargos disponibles.
    /// </summary>
    public async Task<JsonElement> GetRolesAsync()
    {
            var response = await _httpClient.GetAsync($"{_baseUrl}/get_roles");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 6. Get Levels
    /// <summary>
    /// Obtiene la lista de niveles de experiencia disponibles.
    /// </summary>
    public async Task<JsonElement> GetLevelsAsync()
    {
            var response = await _httpClient.GetAsync($"{_baseUrl}/get_levels");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 7. Get Question Bank
    /// <summary>
    /// Obtiene el banco de preguntas filtrado por rol y/o nivel.
    /// </summary>
    /// <param name="role">Rol (opcional)</param>
    /// <param name="level">Nivel (opcional)</param>
    public async Task<JsonElement> GetQuestionBankAsync(string role = null, string level = null)
    {
            var url = $"{_baseUrl}/get_question_bank";
            var query = new List<string>();
            if (!string.IsNullOrEmpty(role)) query.Add($"role={System.Web.HttpUtility.UrlEncode(role)}");
            if (!string.IsNullOrEmpty(level)) query.Add($"level={System.Web.HttpUtility.UrlEncode(level)}");
            if (query.Count > 0) url += "?" + string.Join("&", query);
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 8. Interview History
    /// <summary>
    /// Obtiene el historial de la entrevista en curso para un usuario.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    public async Task<JsonElement> InterviewHistoryAsync(string userId)
    {
            var url = $"{_baseUrl}/interview_history?user_id={System.Web.HttpUtility.UrlEncode(userId)}";
            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 9. Reset Interview
    /// <summary>
    /// Reinicia la sesión de entrevista de un usuario.
    /// </summary>
    /// <param name="userId">ID del usuario</param>
    public async Task<JsonElement> ResetInterviewAsync(string userId)
    {
            var url = $"{_baseUrl}/reset_interview?user_id={System.Web.HttpUtility.UrlEncode(userId)}";
            var response = await _httpClient.PostAsync(url, null);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 10. STT (Speech to Text)
    /// <summary>
    /// Envía audio para transcribirlo a texto (STT).
    /// </summary>
    /// <param name="audioBytes">Audio en bytes</param>
    /// <param name="lang">Idioma (por defecto "es")</param>
    public async Task<JsonElement> SpeechToTextAsync(byte[] audioBytes, string lang = "es")
    {
            var content = new MultipartFormDataContent();
            content.Add(new ByteArrayContent(audioBytes), "audio", "audio.wav");
            content.Add(new StringContent(lang), "lang");
            var response = await _httpClient.PostAsync($"{_baseUrl}/stt", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        // 11. TTS (Text to Speech)
    /// <summary>
    /// Convierte texto a audio (TTS).
    /// </summary>
    /// <param name="text">Texto a sintetizar</param>
    /// <param name="lang">Idioma (por defecto "es")</param>
    public async Task<byte[]> TextToSpeechAsync(string text, string lang = "es")
    {
            var content = new MultipartFormDataContent();
            content.Add(new StringContent(text), "text");
            content.Add(new StringContent(lang), "lang");
            var response = await _httpClient.PostAsync($"{_baseUrl}/tts", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }
    }
}
