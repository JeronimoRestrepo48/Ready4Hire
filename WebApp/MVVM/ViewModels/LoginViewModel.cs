using System.Text.RegularExpressions;

namespace Ready4Hire.MVVM.ViewModels
{
    public class LoginViewModel
    {
        string Email { get; set; }
        string Password { get; set; }
        string Name { get; set; }
        string LastName { get; set; }
        //string Country { get; set; }
        string Job { get; set; }
        List<string> Skills { get; set; }
        List<string> Softskills { get; set; }
        List<string> Interests { get; set; }

        public bool ValidateEmail(string email)
        {
            if (string.IsNullOrWhiteSpace(email))
                return false;

            try
            {
                string pattern = @"^[^@\s]+@[^@\s]+\.[^@\s]+$";

                string strictPattern = @"^(?("")("".+?(?<!\\)""@)|(([0-9a-z]((\.(?!\.))|[-!#\$%&'\*\+/=\?\^`\{\}\|~\w])*)(?<=[0-9a-z])@))" +
                                       @"(?(\[)(\[(\d{1,3}\.){3}\d{1,3}\])|(([0-9a-z][-0-9a-z]*[0-9a-z]*\.)+[a-z0-9][\-a-z0-9]{0,22}[a-z0-9]))$";

                bool isValid = Regex.IsMatch(email, strictPattern, RegexOptions.IgnoreCase, TimeSpan.FromMilliseconds(250));

                if (isValid && email.Contains(".."))
                    return false;

                Email = email;
                return isValid;
            }
            catch (RegexMatchTimeoutException)
            {
                return false;
            }
        }
        public bool ValidatePassword(string password)
        {
            if (string.IsNullOrWhiteSpace(password))
                return false;

            if (password.Contains("<") || password.Contains(">") || password.Contains("'") || password.Contains("\""))
                return false;

            if (password.Length < 8)
                return false;

            bool hasLowercase = Regex.IsMatch(password, "[a-z]");
            bool hasUppercase = Regex.IsMatch(password, "[A-Z]");
            bool hasNumber = Regex.IsMatch(password, "[0-9]");
            bool hasSpecialChar = Regex.IsMatch(password, "[^a-zA-Z0-9]");

            if (!hasLowercase || !hasUppercase || !hasNumber || !hasSpecialChar)
                return false;

            if (Regex.IsMatch(password, "(.)\\1{3,}")) // More than 3 repeated chars
                return false;

            Password = password;
            return true;
        }

        // Verifica que el nombre o apellido no sea vacío y no contenga números ni caracteres especiales.
        public bool ValidateString(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;

            // Solo letras y espacios permitidos
            return Regex.IsMatch(value, @"^[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s]+$");
        }

        //Formatea el nombre o apellido: quita espacios y pone la primera letra en mayúscula y el resto en minúscula.
        private string FormatNameOrLastName(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return string.Empty;

            // Elimina espacios extra y normaliza
            var trimmed = value.Trim();
            if (trimmed.Length == 0)
                return string.Empty;

            // Capitaliza la primera letra, el resto minúsculas
            return char.ToUpper(trimmed[0]) + trimmed.Substring(1).ToLower();
        }
    }
}
