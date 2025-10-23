using System.Text.RegularExpressions;
using System.Web;

namespace Ready4Hire.Services
{
    /// <summary>
    /// Servicio de seguridad para prevenir XSS, CSRF y otras vulnerabilidades
    /// </summary>
    public class SecurityService
    {
        /// <summary>
        /// Sanitiza input para prevenir XSS
        /// </summary>
        public string SanitizeInput(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            // HTML Encode para prevenir XSS
            var sanitized = HttpUtility.HtmlEncode(input);

            // Remover scripts potencialmente peligrosos
            sanitized = Regex.Replace(sanitized, @"<script[^>]*>.*?</script>", "", RegexOptions.IgnoreCase);
            sanitized = Regex.Replace(sanitized, @"javascript:", "", RegexOptions.IgnoreCase);
            sanitized = Regex.Replace(sanitized, @"on\w+\s*=", "", RegexOptions.IgnoreCase);

            return sanitized;
        }

        /// <summary>
        /// Valida email con regex seguro
        /// </summary>
        public bool ValidateEmail(string email)
        {
            if (string.IsNullOrWhiteSpace(email))
                return false;

            try
            {
                var regex = new Regex(@"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");
                return regex.IsMatch(email);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Valida que la contraseña cumpla requisitos de seguridad
        /// </summary>
        public (bool isValid, string message) ValidatePassword(string password)
        {
            if (string.IsNullOrWhiteSpace(password))
                return (false, "La contraseña no puede estar vacía");

            if (password.Length < 8)
                return (false, "La contraseña debe tener al menos 8 caracteres");

            if (!Regex.IsMatch(password, @"[A-Z]"))
                return (false, "La contraseña debe contener al menos una mayúscula");

            if (!Regex.IsMatch(password, @"[a-z]"))
                return (false, "La contraseña debe contener al menos una minúscula");

            if (!Regex.IsMatch(password, @"[0-9]"))
                return (false, "La contraseña debe contener al menos un número");

            return (true, "Contraseña válida");
        }

        /// <summary>
        /// Genera token CSRF
        /// </summary>
        public string GenerateCsrfToken()
        {
            return Guid.NewGuid().ToString("N");
        }

        /// <summary>
        /// Previene SQL Injection limpiando caracteres peligrosos
        /// </summary>
        public string PreventSqlInjection(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            // Remover caracteres peligrosos para SQL
            var dangerous = new[] { "'", "\"", ";", "--", "/*", "*/", "xp_", "sp_", "DROP", "DELETE", "INSERT", "UPDATE" };
            var cleaned = input;

            foreach (var danger in dangerous)
            {
                cleaned = cleaned.Replace(danger, "", StringComparison.OrdinalIgnoreCase);
            }

            return cleaned;
        }

        /// <summary>
        /// Valida que el input no exceda un límite de caracteres
        /// </summary>
        public bool ValidateLength(string input, int maxLength = 1000)
        {
            return !string.IsNullOrEmpty(input) && input.Length <= maxLength;
        }

        /// <summary>
        /// Limpia input permitiendo solo caracteres alfanuméricos y espacios
        /// </summary>
        public string CleanAlphanumeric(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return string.Empty;

            return Regex.Replace(input, @"[^a-zA-Z0-9\s]", "");
        }
    }
}

