using Microsoft.AspNetCore.Components;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.ViewModels;
using Ready4Hire.Services;

namespace Ready4Hire.MVVM.Views
{
    public partial class LoginView : ComponentBase, IDisposable
    {
        [Inject]
        public NavigationManager Navigation { get; set; } = null!;
        
        [Inject]
        private IDbContextFactory<AppDbContext> DbFactory { get; set; } = null!;

        [Inject]
        private AuthService AuthService { get; set; } = null!;

        [Inject]
        private SecurityService SecurityService { get; set; } = null!;

        [Inject]
        private HttpClient HttpClient { get; set; } = null!;

        private LoginViewModel? vm;
        
        private string? errorMessage = null;
        private string email = "";
        private string password = "";
        private bool isLoading = false;
        private bool isRegisterMode = false;

        protected override void OnInitialized()
        {
            vm = new LoginViewModel(HttpClient);
        }

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                // Cerrar cualquier sesión anterior (después del primer render cuando JSInterop está disponible)
                await AuthService.LogoutAsync();
                StateHasChanged();
            }
        }

        private async Task HandleLogin()
        {
            try
            {
                errorMessage = null;
                isLoading = true;

                // Validar inputs vacíos
                if (string.IsNullOrWhiteSpace(email) || string.IsNullOrWhiteSpace(password))
                {
                    errorMessage = "Por favor completa todos los campos";
                    return;
                }

                // Sanitizar y validar email
                email = SecurityService.SanitizeInput(email.Trim());
                
                if (!SecurityService.ValidateEmail(email))
                {
                    errorMessage = "Por favor ingresa un correo electrónico válido";
                    return;
                }

                // Intentar login
                var user = await vm!.Login(email, password);
                
                if (user != null)
                {
                    // Guardar sesión de forma segura
                    await AuthService.LoginAsync(user);
                    
                    // Redireccionar al chat
                    Navigation.NavigateTo("/chat/0", true);
                }
                else
                {
                    // Mensaje más específico
                    errorMessage = "Credenciales incorrectas. Verifica tu correo y contraseña.";
                }
            }
            catch (Exception ex)
            {
                errorMessage = "Error al iniciar sesión. Por favor intenta nuevamente.";
                Console.WriteLine($"Login error: {ex.Message}");
            }
            finally
            {
                isLoading = false;
                StateHasChanged();
            }
        }

        private void ShowRegister()
        {
            isRegisterMode = true;
            errorMessage = null;
            Step = 1;
        }

        private void ShowLogin()
        {
            isRegisterMode = false;
            errorMessage = null;
            Step = 1;
            ResetRegistration();
        }

        private string GetStepTitle()
        {
            return Step switch
            {
                1 => "Crea tu cuenta",
                2 => "Tu información personal",
                3 => "Habilidades Técnicas",
                4 => "Habilidades Blandas",
                5 => "Intereses Profesionales",
                _ => "Registro"
            };
        }

        private async Task LoginWithGoogle()
        {
            errorMessage = "La autenticación con Google estará disponible próximamente";
            await Task.CompletedTask;
        }

        private async Task LoginWithMicrosoft()
        {
            errorMessage = "La autenticación con Microsoft estará disponible próximamente";
            await Task.CompletedTask;
        }

        private async Task CompleteRegistration()
        {
            try
            {
                isLoading = true;
                errorMessage = null;

                Console.WriteLine($"[REGISTRO] Iniciando registro para: {registerEmail}");

                // Sanitizar todos los inputs
                registerName = SecurityService.SanitizeInput(registerName);
                registerLastName = SecurityService.SanitizeInput(registerLastName);
                registerEmail = SecurityService.SanitizeInput(registerEmail);
                registerCountry = SecurityService.SanitizeInput(registerCountry);
                registerJob = SecurityService.SanitizeInput(registerJob);

                // Validar longitud de inputs
                if (!SecurityService.ValidateLength(registerName, 100) ||
                    !SecurityService.ValidateLength(registerLastName, 100) ||
                    !SecurityService.ValidateLength(registerJob, 200))
                {
                    errorMessage = "Uno o más campos exceden el límite de caracteres";
                    return;
                }

                Console.WriteLine($"[REGISTRO] Guardando usuario en BD...");

                // Registrar usuario
                await vm!.FinishRegistration(
                    registerName, 
                    registerLastName, 
                    registerJob, 
                    registerCountry,
                    selectedHardSkills, 
                    selectedSoftSkills, 
                    selectedInterests
                );

                Console.WriteLine($"[REGISTRO] Usuario guardado. Intentando login automático...");

                // Login automático después del registro (usando la misma instancia de VM)
                var user = await vm!.Login(registerEmail, registerPassword);
                
                if (user != null)
                {
                    Console.WriteLine($"[REGISTRO] Login exitoso. Usuario ID: {user.Id}");
                    
                    // Guardar sesión de forma segura
                    await AuthService.LoginAsync(user);
                    
                    Console.WriteLine($"[REGISTRO] Sesión guardada. Redirigiendo al chat...");
                    
                    // Redireccionar al chat
                    Navigation.NavigateTo("/chat/0", true);
                }
                else
                {
                    Console.WriteLine($"[REGISTRO] Login automático falló. Mostrando pantalla de login.");
                    errorMessage = "Registro exitoso. Por favor inicia sesión.";
                    ShowLogin();
                }
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al registrar: {ex.Message}";
                Console.WriteLine($"[REGISTRO ERROR] {ex.Message}");
                Console.WriteLine($"[REGISTRO ERROR] StackTrace: {ex.StackTrace}");
            }
            finally
            {
                isLoading = false;
                StateHasChanged();
            }
        }

        private bool showRegisterModal = false;
        private int Step = 1;

        private string hardSkillSearch = "";
        private string softSkillSearch = "";
        private string interestSearch = "";

        // Step 1 registration fields
        private string registerEmail = "";
        private string registerPassword = "";
        private string registerConfirmPassword = "";

        // Step 2 registration fields
        private string registerName = "";
        private string registerLastName = "";
        private string registerCountry = "";
        private string registerJob = "";

        // Validation state step 1
        private bool isEmailInvalid = false;
        private bool isPasswordInvalid = false;
        private bool isConfirmPasswordInvalid = false;

        // Validation state step 2
        private bool isNameInvalid = false;
        private bool isLastNameInvalid = false;
        private bool isCountryInvalid = false;
        private bool isJobInvalid = false;

        // Validation state step 3
        private bool isHardskillsInvalid = false;
        private bool isSoftskillsInvalid = false;
        private bool isInterestsInvalid = false;

        #region Listas de habilidades e intereses
        private List<string> hardSkills = new()
        {
            "C#", "Java", "SQL", "JavaScript", "Python", "Docker", "Kubernetes",
            "Go", "TypeScript", "Rust", "Scala", "Ruby", "Swift", "React",
            "Angular", "Node.js", "AWS", "Azure", "Google Cloud", "PostgreSQL",
            "MongoDB", "GraphQL", "ElasticSearch", "Terraform", "CI/CD", "Jenkins",
            "Git", "Bash", "PowerShell", "Unity", "Unreal Engine", "F#", "Objective-C",
            "PHP", "Laravel", "Django", "Flask", "Spring Boot", "ASP.NET Core",
            "Hibernate", "Redis", "RabbitMQ", "Kafka", "Microservices",
            "Data Engineering", "Machine Learning", "Computer Vision", "Hadoop",
            "Spark", "Airflow"
        };
        private List<string> softSkills = new()
        {
            "Comunicación", "Trabajo en equipo", "Liderazgo", "Adaptabilidad", "Pensamiento crítico",
            "Creatividad", "Gestión del tiempo", "Resolución de problemas", "Empatía",
            "Pensamiento estratégico", "Comunicación intercultural", "Negociación",
            "Gestión del estrés", "Capacidad de aprendizaje", "Orientación al detalle",
            "Inteligencia emocional", "Gestión de conflictos", "Proactividad",
            "Pensamiento analítico", "Autoconfianza", "Toma de decisiones", "Mentoría",
            "Escucha activa", "Colaboración", "Gestión del cambio", "Planificación estratégica",
            "Resiliencia", "Responsabilidad", "Orientación al cliente", "Networking",
            "Coaching", "Delegación", "Capacidad de síntesis", "Comunicación escrita",
            "Oratoria", "Gestión de proyectos", "Innovación", "Flexibilidad", "Autogestión",
            "Mentalidad de crecimiento", "Disciplina", "Motivación", "Diplomacia",
            "Ética profesional", "Perseverancia", "Cultura de feedback", "Pensamiento sistémico",
            "Orientación a resultados", "Gestión de prioridades", "Aprendizaje continuo"
        };
        private List<string> interests = new()
        {
            "Inteligencia Artificial", "Desarrollo Web", "Videojuegos", "Ciberseguridad", "Cloud Computing",
            "Realidad Aumentada", "Realidad Virtual", "Blockchain", "Data Science", "Internet of Things",
            "Machine Learning", "Robótica", "Automatización", "FinTech", "EdTech", "Análisis de Datos",
            "DevOps", "Emprendimiento tecnológico", "Computación Cuántica", "Bioinformática",
            "Diseño de UX/UI", "Desarrollo Mobile", "Sistemas Embebidos", "Big Data", "Agrotech",
            "Greentech", "Realidad Mixta", "Ciencia Ciudadana", "Economía Digital", "Smart Cities",
            "Automoción autónoma", "Energías Renovables", "eSports", "Realidad Extendida",
            "Computación Edge", "Seguridad de Datos", "Criptografía", "Computación Distribuida",
            "Ingeniería de Datos", "Narrativa Interactiva", "Impresión 3D", "Educación en Línea",
            "Microservicios", "Tecnología Sanitaria", "Automatización de Procesos", "Gobernanza de Datos",
            "Metaverso", "Nanotecnología", "Tecnologías Vestibles", "Analítica Predictiva"
        };
        #endregion

        private List<string> filteredHardSkills = new();
        private List<string> filteredSoftSkills = new();
        private List<string> filteredInterests = new();

        private List<string> selectedHardSkills = new();
        private List<string> selectedSoftSkills = new();
        private List<string> selectedInterests = new();

        private List<string> availableHardSkills => hardSkills;
        private List<string> availableSoftSkills => softSkills;
        private List<string> availableInterests => interests;

        void ShowRegisterModal()
        {
            showRegisterModal = true;
            Step = 1;
            ResetSearch();
            ResetValidation();
            ResetRegistration();
        }

        void ResetRegistration()
        {
            registerEmail = "";
            registerPassword = "";
            registerConfirmPassword = "";
            registerName = "";
            registerLastName = "";
            registerCountry = "";
            registerJob = "";
            selectedHardSkills.Clear();
            selectedSoftSkills.Clear();
            selectedInterests.Clear();
        }

        void HideRegisterModal()
        {
            showRegisterModal = false;
            Step = 1;
            ResetSearch();
            ResetValidation();
        }

        void ResetValidation()
        {
            isEmailInvalid = false;
            isPasswordInvalid = false;
            isConfirmPasswordInvalid = false;
            isNameInvalid = false;
            isLastNameInvalid = false;
            isCountryInvalid = false;
            isJobInvalid = false;
            isHardskillsInvalid = false;
            isSoftskillsInvalid = false;
            isInterestsInvalid = false;
        }

        #region MetodosPaso3

        void ResetSearch()
        {
            hardSkillSearch = "";
            softSkillSearch = "";
            interestSearch = "";
            filteredHardSkills.Clear();
            filteredSoftSkills.Clear();
            filteredInterests.Clear();
        }

        void FilterHardSkills(ChangeEventArgs? e = null)
        {
            if (e != null)
                hardSkillSearch = e.Value?.ToString() ?? "";
            
            if (string.IsNullOrEmpty(hardSkillSearch))
            {
                filteredHardSkills.Clear(); // No mostrar todas si está vacío
            }
            else
            {
                // Mostrar solo coincidencias mientras escribe (autocompletado)
                filteredHardSkills = hardSkills
                    .Where(s => !selectedHardSkills.Contains(s) && 
                               s.Contains(hardSkillSearch, StringComparison.OrdinalIgnoreCase))
                    .Take(10) // Limitar a 10 resultados para mejor UX
                    .ToList();
            }
            StateHasChanged();
        }

        void FilterSoftSkills(ChangeEventArgs? e = null)
        {
            if (e != null)
                softSkillSearch = e.Value?.ToString() ?? "";
            
            if (string.IsNullOrEmpty(softSkillSearch))
            {
                filteredSoftSkills.Clear(); // No mostrar todas si está vacío
            }
            else
            {
                // Mostrar solo coincidencias mientras escribe (autocompletado)
                filteredSoftSkills = softSkills
                    .Where(s => !selectedSoftSkills.Contains(s) && 
                               s.Contains(softSkillSearch, StringComparison.OrdinalIgnoreCase))
                    .Take(10) // Limitar a 10 resultados
                    .ToList();
            }
            StateHasChanged();
        }

        void FilterInterests(ChangeEventArgs? e = null)
        {
            if (e != null)
                interestSearch = e.Value?.ToString() ?? "";
            
            if (string.IsNullOrEmpty(interestSearch))
            {
                filteredInterests.Clear(); // No mostrar todos si está vacío
            }
            else
            {
                // Mostrar solo coincidencias mientras escribe (autocompletado)
                filteredInterests = interests
                    .Where(i => !selectedInterests.Contains(i) && 
                               i.Contains(interestSearch, StringComparison.OrdinalIgnoreCase))
                    .Take(10) // Limitar a 10 resultados
                    .ToList();
            }
            StateHasChanged();
        }

        void ToggleHardSkill(string skill)
        {
            if (selectedHardSkills.Contains(skill))
                selectedHardSkills.Remove(skill);
            else
                selectedHardSkills.Add(skill);
        }

        void AddHardSkill(string skill)
        {
            selectedHardSkills.Add(skill);
            hardSkillSearch = "";
            filteredHardSkills.Clear();
        }

        void RemoveHardSkill(string skill)
        {
            selectedHardSkills.Remove(skill);
        }

        void ToggleSoftSkill(string skill)
        {
            if (selectedSoftSkills.Contains(skill))
                selectedSoftSkills.Remove(skill);
            else
                selectedSoftSkills.Add(skill);
        }

        void AddSoftSkill(string skill)
        {
            selectedSoftSkills.Add(skill);
            softSkillSearch = "";
            filteredSoftSkills.Clear();
        }

        void RemoveSoftSkill(string skill)
        {
            selectedSoftSkills.Remove(skill);
        }

        void ToggleInterest(string interest)
        {
            if (selectedInterests.Contains(interest))
                selectedInterests.Remove(interest);
            else
                selectedInterests.Add(interest);
        }

        void AddInterest(string interest)
        {
            selectedInterests.Add(interest);
            interestSearch = "";
            filteredInterests.Clear();
        }

        void RemoveInterest(string interest)
        {
            selectedInterests.Remove(interest);
        }

        void PreviousStep()
        {
            if (Step > 1)
                Step--;
        }

        async Task NextStep()
        {
            if (Step == 1)
            {
                // Validate email and password
                if (string.IsNullOrWhiteSpace(registerEmail) || string.IsNullOrWhiteSpace(registerPassword))
                {
                    errorMessage = "Por favor completa todos los campos";
                    return;
                }

                if (!SecurityService.ValidateEmail(registerEmail))
                {
                    errorMessage = "Por favor ingresa un correo electrónico válido";
                    return;
                }

                var passwordValidation = SecurityService.ValidatePassword(registerPassword);
                if (!passwordValidation.isValid)
                {
                    errorMessage = passwordValidation.message;
                    return;
                }

                await vm!.SaveEmailAndPassword(registerEmail, registerPassword);
            }
            else if (Step == 2)
            {
                // Validate personal info
                if (string.IsNullOrWhiteSpace(registerName) || string.IsNullOrWhiteSpace(registerLastName) ||
                    string.IsNullOrWhiteSpace(registerJob) || string.IsNullOrWhiteSpace(registerCountry))
                {
                    errorMessage = "Por favor completa todos los campos";
                    return;
                }
            }

            if (Step < 5)
            {
                Step++;
                errorMessage = null;
            }
        }
        #endregion

        public void Dispose()
        {
            // No hay recursos para liberar
        }
    }
}
