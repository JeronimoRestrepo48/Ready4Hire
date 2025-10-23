using Microsoft.AspNetCore.Components.Server.ProtectedBrowserStorage;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Services
{
    /// <summary>
    /// Servicio de autenticación seguro con gestión de sesión
    /// </summary>
    public class AuthService
    {
        private readonly ProtectedSessionStorage _sessionStorage;
        private const string USER_KEY = "current_user";
        private const string SESSION_TOKEN_KEY = "session_token";

        public AuthService(ProtectedSessionStorage sessionStorage)
        {
            _sessionStorage = sessionStorage;
        }

        /// <summary>
        /// Inicia sesión y guarda el usuario en la sesión protegida
        /// </summary>
        public async Task LoginAsync(User user)
        {
            // Generar token de sesión único
            var sessionToken = Guid.NewGuid().ToString();
            
            try
            {
                // Guardar en almacenamiento protegido del servidor
                await _sessionStorage.SetAsync(USER_KEY, user);
                await _sessionStorage.SetAsync(SESSION_TOKEN_KEY, sessionToken);
            }
            catch (InvalidOperationException)
            {
                // Ignorar si JSInterop no está disponible aún
            }
        }

        /// <summary>
        /// Cierra la sesión del usuario actual
        /// </summary>
        public async Task LogoutAsync()
        {
            try
            {
                await _sessionStorage.DeleteAsync(USER_KEY);
                await _sessionStorage.DeleteAsync(SESSION_TOKEN_KEY);
            }
            catch (InvalidOperationException)
            {
                // Ignorar si JSInterop no está disponible
            }
        }

        /// <summary>
        /// Obtiene el usuario actual de la sesión
        /// </summary>
        public async Task<User?> GetCurrentUserAsync()
        {
            try
            {
                var result = await _sessionStorage.GetAsync<User>(USER_KEY);
                return result.Success ? result.Value : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Verifica si hay un usuario autenticado
        /// </summary>
        public async Task<bool> IsAuthenticatedAsync()
        {
            try
            {
                var user = await GetCurrentUserAsync();
                return user != null;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Verifica el token de sesión
        /// </summary>
        public async Task<bool> ValidateSessionAsync()
        {
            try
            {
                var tokenResult = await _sessionStorage.GetAsync<string>(SESSION_TOKEN_KEY);
                var userResult = await _sessionStorage.GetAsync<User>(USER_KEY);
                
                return tokenResult.Success && userResult.Success && 
                       !string.IsNullOrEmpty(tokenResult.Value);
            }
            catch
            {
                return false;
            }
        }
    }
}

