using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Ready4Hire.Services
{
    /// <summary>
    /// Servicio optimizado para comunicación en tiempo real con el backend
    /// Implementa streaming de respuestas para reducir la latencia percibida
    /// </summary>
    public class StreamingApiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public StreamingApiService(HttpClient httpClient, IConfiguration configuration)
        {
            _httpClient = httpClient;
            _baseUrl = configuration["PythonApi:BaseUrl"] ?? "http://localhost:8001";
        }

        /// <summary>
        /// Procesa respuesta del candidato con streaming
        /// Permite mostrar evaluación en tiempo real mientras se genera
        /// </summary>
        public async IAsyncEnumerable<EvaluationChunk> ProcessAnswerStream(
            string interviewId,
            string answerText,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var requestBody = new
            {
                answer_text = answerText,
                timestamp = DateTime.UtcNow
            };

            var content = new StringContent(
                JsonSerializer.Serialize(requestBody),
                Encoding.UTF8,
                "application/json"
            );

            var request = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/api/v2/interviews/{interviewId}/answers/stream")
            {
                Content = content
            };

            using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            
            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException($"Error processing answer: {response.StatusCode}");
            }

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line)) continue;

                // Parse NDJSON (newline-delimited JSON)
                var chunk = JsonSerializer.Deserialize<EvaluationChunk>(line);
                if (chunk != null)
                {
                    yield return chunk;
                }
            }
        }

        /// <summary>
        /// Optimistic update: Añade mensaje del usuario inmediatamente
        /// Luego obtiene respuesta del servidor
        /// </summary>
        public async Task<EvaluationResponse> ProcessAnswerWithOptimisticUpdate(
            string interviewId,
            string answerText,
            Action<string> onUserMessageAdded)
        {
            // 1. Mostrar mensaje del usuario inmediatamente (optimistic)
            onUserMessageAdded?.Invoke(answerText);

            // 2. Llamar al backend
            var requestBody = new
            {
                answer_text = answerText,
                timestamp = DateTime.UtcNow
            };

            var content = new StringContent(
                JsonSerializer.Serialize(requestBody),
                Encoding.UTF8,
                "application/json"
            );

            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/api/v2/interviews/{interviewId}/answers",
                content
            );

            response.EnsureSuccessStatusCode();
            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<EvaluationResponse>(responseContent);
        }

        /// <summary>
        /// Pre-carga siguiente pregunta en background
        /// Mejora la percepción de velocidad
        /// </summary>
        public async Task<Question> PrefetchNextQuestion(string interviewId)
        {
            try
            {
                var response = await _httpClient.GetAsync(
                    $"{_baseUrl}/api/v2/interviews/{interviewId}/next-question"
                );

                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<Question>(content);
                }
            }
            catch
            {
                // Silently fail, no es crítico
            }

            return null;
        }
    }

    public class EvaluationChunk
    {
        public string Type { get; set; } // "score", "feedback", "next_question", "complete"
        public object Data { get; set; }
        public bool IsFinal { get; set; }
    }

    public class EvaluationResponse
    {
        public EvaluationData Evaluation { get; set; }
        public string Feedback { get; set; }
        public EmotionData Emotion { get; set; }
        public Question NextQuestion { get; set; }
        public ProgressData Progress { get; set; }
        public string InterviewStatus { get; set; }
        public bool InterviewCompleted { get; set; }
    }

    public class EvaluationData
    {
        public float Score { get; set; }
        public bool IsCorrect { get; set; }
        public string Feedback { get; set; }
        public List<string> Strengths { get; set; }
        public List<string> Improvements { get; set; }
    }

    public class EmotionData
    {
        public string Emotion { get; set; }
        public float Confidence { get; set; }
    }

    public class Question
    {
        public string Id { get; set; }
        public string Text { get; set; }
        public string Category { get; set; }
        public string Difficulty { get; set; }
        public string Topic { get; set; }
    }

    public class ProgressData
    {
        public int QuestionsCompleted { get; set; }
        public int TotalQuestions { get; set; }
        public float Percentage { get; set; }
    }
}

