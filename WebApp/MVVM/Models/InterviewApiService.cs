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

        // ============================================================================
        // API V2 ENDPOINTS - Flujo Conversacional con Fases de Contexto
        // ============================================================================

        /// <summary>
        /// [V2] Inicia una nueva entrevista con fase de contexto.
        /// Retorna la primera pregunta de contexto para conocer al candidato.
        /// </summary>
        /// <param name="userId">ID del usuario</param>
        /// <param name="role">Rol/posición (ej: Backend Developer)</param>
        /// <param name="category">Categoría: technical o soft_skills</param>
        /// <param name="difficulty">Dificultad: junior, mid, senior</param>
        public async Task<JsonElement> StartInterviewV2Async(string userId, string role, string category, string difficulty)
        {
            try
            {
                var payload = new 
                { 
                    user_id = userId, 
                    role = role, 
                    category = category, 
                    difficulty = difficulty 
                };
                
                var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/v2/interviews", payload);
                
                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    throw new HttpRequestException($"API Error ({response.StatusCode}): {errorContent}");
                }
                
                var jsonContent = await response.Content.ReadAsStringAsync();
                
                // Verificar que la respuesta no esté vacía y sea JSON válido
                if (string.IsNullOrWhiteSpace(jsonContent))
                {
                    throw new InvalidOperationException("La respuesta del servidor está vacía");
                }
                
                // Verificar que no empiece con caracteres inválidos
                var trimmedContent = jsonContent.TrimStart();
                if (trimmedContent.StartsWith("#"))
                {
                    if (trimmedContent.Contains("python_gc_objects") || trimmedContent.Contains("HELP") || trimmedContent.Contains("TYPE"))
                    {
                        throw new InvalidOperationException("El servidor está devolviendo métricas de Prometheus en lugar de la API. Verifica que la URL base sea correcta y que el backend de entrevistas esté ejecutándose en el puerto correcto.");
                    }
                    else
                    {
                        throw new InvalidOperationException($"Respuesta inválida del servidor: {jsonContent.Substring(0, Math.Min(100, jsonContent.Length))}...");
                    }
                }
                if (trimmedContent.StartsWith("<"))
                {
                    throw new InvalidOperationException("El servidor está devolviendo HTML en lugar de JSON. Verifica que la URL de la API sea correcta.");
                }
                
                return JsonSerializer.Deserialize<JsonElement>(jsonContent);
            }
            catch (HttpRequestException)
            {
                // Re-throw HTTP errors as-is
                throw;
            }
            catch (TaskCanceledException)
            {
                throw new TimeoutException("La conexión con el servidor ha expirado. Verifica que el backend esté ejecutándose.");
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException($"Error al parsear la respuesta JSON: {ex.Message}");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error inesperado al comunicarse con el servidor: {ex.Message}");
            }
        }

        /// <summary>
        /// [V2] Procesa una respuesta del candidato.
        /// Maneja tanto preguntas de contexto como preguntas técnicas/soft skills.
        /// Incluye detección de emoción, evaluación, feedback y siguiente pregunta.
        /// </summary>
        /// <param name="interviewId">ID de la entrevista</param>
        /// <param name="answer">Respuesta del candidato</param>
        /// <param name="timeTaken">Tiempo en segundos (opcional)</param>
        public async Task<JsonElement> ProcessAnswerV2Async(string interviewId, string answer, int? timeTaken = null)
        {
            var payload = new 
            { 
                answer = answer, 
                time_taken = timeTaken 
            };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/v2/interviews/{interviewId}/answers", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Finaliza la entrevista y genera resumen completo.
        /// </summary>
        /// <param name="interviewId">ID de la entrevista</param>
        public async Task<JsonElement> EndInterviewV2Async(string interviewId)
        {
            var response = await _httpClient.PostAsync($"{_baseUrl}/api/v2/interviews/{interviewId}/end", null);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Obtiene el estado de la entrevista activa de un usuario.
        /// Usado para restaurar el estado cuando el usuario vuelve a la página.
        /// </summary>
        /// <param name="userId">ID del usuario</param>
        public async Task<JsonElement> GetActiveInterviewAsync(string userId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/active/{userId}");
            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                // No hay entrevista activa, retornar JSON vacío
                return JsonSerializer.Deserialize<JsonElement>("{}");
            }
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Health check del sistema.
        /// </summary>
        public async Task<JsonElement> HealthCheckV2Async()
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/health");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Obtiene todas las entrevistas completadas de un usuario.
        /// Incluye información básica de reportes y certificados.
        /// </summary>
        /// <param name="userId">ID del usuario</param>
        /// <param name="limit">Límite de resultados (por defecto 10)</param>
        public async Task<JsonElement> GetCompletedInterviewsAsync(string userId, int limit = 10)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/user/{userId}/completed?limit={limit}");
            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                return JsonSerializer.Deserialize<JsonElement>("{\"interviews\": [], \"total\": 0}");
            }
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Obtiene el reporte completo de una entrevista completada.
        /// </summary>
        /// <param name="interviewId">ID de la entrevista</param>
        public async Task<JsonElement> GetInterviewReportAsync(string interviewId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/{interviewId}/report");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Obtiene el certificado de una entrevista completada.
        /// </summary>
        /// <param name="interviewId">ID de la entrevista</param>
        /// <param name="format">Formato: json, svg, o pdf (por defecto json)</param>
        public async Task<JsonElement> GetInterviewCertificateAsync(string interviewId, string format = "json")
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/{interviewId}/certificate?format={format}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// [V2] Descarga el certificado en formato SVG o PDF.
        /// </summary>
        /// <param name="interviewId">ID de la entrevista</param>
        /// <param name="format">Formato: svg o pdf</param>
        public async Task<byte[]> DownloadCertificateAsync(string interviewId, string format = "svg")
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/{interviewId}/certificate?format={format}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }

        // ============================================================================
        // API V1 ENDPOINTS - Legacy (mantener para compatibilidad)
        // ============================================================================

        // 1. Start Interview
    /// <summary>
    /// [V1 LEGACY] Inicia una nueva entrevista para un usuario.
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
    public async Task<JsonElement> GetQuestionBankAsync(string? role = null, string? level = null)
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
            content.Add(new ByteArrayContent(audioBytes), "audio_file", "audio.wav");
            content.Add(new StringContent(lang), "language");
            var response = await _httpClient.PostAsync($"{_baseUrl}/api/v2/audio/speech-to-text", content);
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
            var payload = new 
            {
                text = text,
                language = lang,
                rate = 150,
                volume = 1.0f,
                output_format = "mp3"
            };
            
            var jsonContent = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = await _httpClient.PostAsync($"{_baseUrl}/api/v2/audio/text-to-speech-bytes", jsonContent);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }

        // ============================================================================
        // USER SETTINGS & DATA EXPORT ENDPOINTS
        // ============================================================================

        /// <summary>
        /// Obtiene todas las entrevistas completadas del usuario para exportación
        /// </summary>
        /// <param name="userId">ID del usuario</param>
        public async Task<JsonElement> GetAllUserInterviewsAsync(string userId)
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/api/v2/interviews/user/{userId}/completed?limit=1000");
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadFromJsonAsync<JsonElement>();
            }
            catch (HttpRequestException ex)
            {
                throw new Exception($"Error al obtener entrevistas: {ex.Message}", ex);
            }
        }
    }
}
