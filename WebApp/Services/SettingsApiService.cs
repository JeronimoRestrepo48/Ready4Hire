using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text;

namespace Ready4Hire.Services
{
    /// <summary>
    /// Servicio para interactuar con los endpoints de configuración del backend
    /// </summary>
    public class SettingsApiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public SettingsApiService(HttpClient httpClient, IConfiguration configuration)
        {
            _httpClient = httpClient;
            // El backend de WebApp corre en el mismo servidor, así que usamos la URL base relativa
            // En Blazor Server, las llamadas al mismo servidor se hacen con URL relativa
            _baseUrl = "";
        }

        /// <summary>
        /// Cambia la contraseña del usuario
        /// </summary>
        public async Task<JsonElement> ChangePasswordAsync(string email, string currentPassword, string newPassword)
        {
            var payload = new
            {
                email = email,
                currentPassword = currentPassword,
                newPassword = newPassword
            };

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/settings/change-password", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Obtiene las preferencias de notificaciones
        /// </summary>
        public async Task<JsonElement> GetNotificationsAsync(int userId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/settings/notifications/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Actualiza las preferencias de notificaciones
        /// </summary>
        public async Task<JsonElement> UpdateNotificationsAsync(int userId, bool emailNotifications, bool achievementNotifications, bool interviewReminders)
        {
            var payload = new
            {
                userId = userId,
                emailNotifications = emailNotifications,
                achievementNotifications = achievementNotifications,
                interviewReminders = interviewReminders
            };

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/settings/notifications", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Obtiene las preferencias de privacidad
        /// </summary>
        public async Task<JsonElement> GetPrivacyAsync(int userId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/settings/privacy/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Actualiza las preferencias de privacidad
        /// </summary>
        public async Task<JsonElement> UpdatePrivacyAsync(int userId, bool showProfilePublic, bool showStatsPublic, bool allowDataSharing)
        {
            var payload = new
            {
                userId = userId,
                showProfilePublic = showProfilePublic,
                showStatsPublic = showStatsPublic,
                allowDataSharing = allowDataSharing
            };

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/settings/privacy", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Obtiene las preferencias de idioma
        /// </summary>
        public async Task<JsonElement> GetLanguageAsync(int userId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/settings/language/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Actualiza las preferencias de idioma
        /// </summary>
        public async Task<JsonElement> UpdateLanguageAsync(int userId, string language, string region, string? timeZone = null)
        {
            var payload = new
            {
                userId = userId,
                language = language,
                region = region,
                timeZone = timeZone
            };

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/settings/language", payload);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Elimina todos los datos del usuario
        /// </summary>
        public async Task<JsonElement> DeleteAllDataAsync(int userId)
        {
            var response = await _httpClient.DeleteAsync($"{_baseUrl}/api/settings/delete-all-data/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Elimina la cuenta del usuario permanentemente
        /// </summary>
        public async Task<JsonElement> DeleteAccountAsync(int userId)
        {
            var response = await _httpClient.DeleteAsync($"{_baseUrl}/api/settings/delete-account/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }

        /// <summary>
        /// Obtiene todas las preferencias del usuario
        /// </summary>
        public async Task<JsonElement> GetAllSettingsAsync(int userId)
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/settings/all/{userId}");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<JsonElement>();
        }
    }
}

