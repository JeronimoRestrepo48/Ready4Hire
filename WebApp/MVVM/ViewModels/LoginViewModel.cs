using Ready4Hire.MVVM.Models;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text.Json;

namespace Ready4Hire.MVVM.ViewModels
{
    public class LoginViewModel
    {
        private readonly HttpClient _httpClient;
        
        // Campos temporales para el registro multi-paso
        private string? _tempEmail;
        private string? _tempPassword;

        public LoginViewModel(HttpClient httpClient)
        {
            _httpClient = httpClient;
            _httpClient.BaseAddress = new Uri("http://localhost:5214");
        }

        /// <summary>
        /// Guarda email y contraseña temporalmente para el proceso de registro
        /// </summary>
        public async Task SaveEmailAndPassword(string email, string password)
        {
            _tempEmail = email;
            _tempPassword = password;
            await Task.CompletedTask;
        }

        /// <summary>
        /// Autentica un usuario con email y contraseña mediante la API
        /// </summary>
        public async Task<User?> Login(string email, string password)
        {
            try
            {
                var response = await _httpClient.PostAsJsonAsync("/api/auth/login", new
                {
                    email = email,
                    password = password
                });

                if (!response.IsSuccessStatusCode)
                    return null;

                var result = await response.Content.ReadFromJsonAsync<JsonElement>();
                
                if (!result.GetProperty("success").GetBoolean())
                    return null;

                var userJson = result.GetProperty("user");
                
                return new User
                {
                    Id = userJson.GetProperty("id").GetInt32(),
                    Email = userJson.GetProperty("email").GetString() ?? "",
                    Name = userJson.GetProperty("name").GetString() ?? "",
                    LastName = userJson.GetProperty("lastName").GetString() ?? "",
                    Country = userJson.GetProperty("country").GetString() ?? "",
                    Job = userJson.GetProperty("job").GetString() ?? "",
                    Skills = JsonSerializer.Deserialize<List<string>>(userJson.GetProperty("skills").GetRawText()) ?? new(),
                    Softskills = JsonSerializer.Deserialize<List<string>>(userJson.GetProperty("softskills").GetRawText()) ?? new(),
                    Interests = JsonSerializer.Deserialize<List<string>>(userJson.GetProperty("interests").GetRawText()) ?? new()
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Completa el registro del usuario con toda la información mediante la API
        /// </summary>
        public async Task FinishRegistration(
            string name, 
            string lastName, 
            string job, 
            string country,
            List<string> skills, 
            List<string> softskills, 
            List<string> interests)
        {
            if (string.IsNullOrEmpty(_tempEmail) || string.IsNullOrEmpty(_tempPassword))
            {
                throw new InvalidOperationException("Email y contraseña no han sido guardados. Llama a SaveEmailAndPassword primero.");
            }

            var response = await _httpClient.PostAsJsonAsync("/api/auth/register", new
            {
                email = _tempEmail,
                password = _tempPassword,
                name = name,
                lastName = lastName,
                job = job,
                country = country,
                skills = skills,
                softskills = softskills,
                interests = interests
            });

            response.EnsureSuccessStatusCode();

            // Limpiar campos temporales
            _tempEmail = null;
            _tempPassword = null;
        }
    }
}
