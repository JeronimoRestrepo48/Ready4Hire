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

        private LoginViewModel? vm;
        private AppDbContext? currentDb;
        
        private string? loginError = null;
        private string email = "";
        private string password = "";

        protected override async Task OnInitializedAsync()
        {
            // Cerrar cualquier sesión anterior
            await AuthService.LogoutAsync();
            
            currentDb = await DbFactory.CreateDbContextAsync();
            vm = new LoginViewModel(currentDb);
        }

        private async Task HandleLogin()
        {
            try
            {
                loginError = null;

                // Validar y sanitizar inputs
                email = SecurityService.SanitizeInput(email);
                
                if (!SecurityService.ValidateEmail(email))
                {
                    loginError = "Email inválido";
                    return;
                }

                if (string.IsNullOrWhiteSpace(password))
                {
                    loginError = "La contraseña no puede estar vacía";
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
                    loginError = "Email o contraseña incorrectos";
                }
            }
            catch (Exception ex)
            {
                loginError = "Error al iniciar sesión. Intenta nuevamente.";
                Console.WriteLine($"Login error: {ex.Message}");
            }
        }

        private async Task CompleteRegistration()
        {
            try
            {
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
                    loginError = "Uno o más campos exceden el límite de caracteres";
                    return;
                }

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

                // Login automático después del registro
                var user = await vm.Login(registerEmail, registerPassword);
                if (user != null)
                {
                    await AuthService.LoginAsync(user);
                    Navigation.NavigateTo("/chat/0", true);
                }
            }
            catch (Exception ex)
            {
                loginError = "Error al registrar. Intenta nuevamente.";
                Console.WriteLine($"Registration error: {ex.Message}");
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

        void ShowRegisterModal()
        {
            showRegisterModal = true;
            Step = 1;
            ResetSearch();
            ResetValidation();

            registerEmail = "";
            registerPassword = "";
            registerConfirmPassword = "";
            registerName = "";
            registerLastName = "";
            registerCountry = "Colombia";
            registerJob = "";
        }

        async void NextStep()
        {
            if (Step == 1)
            {
                // Validate email and password
                isEmailInvalid = !vm.ValidateEmail(registerEmail);
                isPasswordInvalid = !vm.ValidatePassword(registerPassword);
                isConfirmPasswordInvalid = registerPassword != registerConfirmPassword;

                if (isEmailInvalid || isPasswordInvalid || isConfirmPasswordInvalid)
                    return;
            }
            else if (Step == 2)
            {
                // Validate name and last name
                isNameInvalid = !vm.ValidateString(registerName);
                isLastNameInvalid = !vm.ValidateString(registerLastName);
                isCountryInvalid = !vm.ValidateString(registerCountry);
                isJobInvalid = !vm.ValidateString(registerJob);

                if (isNameInvalid || isLastNameInvalid || isCountryInvalid || isJobInvalid)
                    return;
            }
            else if (Step == 3)
            {
                // Validate skills and interests
                isHardskillsInvalid = selectedHardSkills.Count == 0;
                isSoftskillsInvalid = selectedSoftSkills.Count == 0;
                isInterestsInvalid = selectedInterests.Count == 0;

                // Prevent continue if any required selection is missing
                if (isHardskillsInvalid || isSoftskillsInvalid || isInterestsInvalid)
                    return;
                else
                {
                    await vm.FinishRegistration(registerName, registerLastName, registerJob, registerCountry,
                        selectedHardSkills, selectedSoftSkills, selectedInterests);

                    Navigation.NavigateTo("/chat/0");
                }
            }

            if (Step < 3)
                Step++;

            ResetSearch();
            ResetValidation();
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

        void FilterHardSkills(ChangeEventArgs e)
        {
            hardSkillSearch = e.Value?.ToString() ?? "";
            filteredHardSkills = hardSkills
                .Where(s => s.Contains(hardSkillSearch, StringComparison.OrdinalIgnoreCase) && !selectedHardSkills.Contains(s))
                .ToList();
        }

        void FilterSoftSkills(ChangeEventArgs e)
        {
            softSkillSearch = e.Value?.ToString() ?? "";
            filteredSoftSkills = softSkills
                .Where(s => s.Contains(softSkillSearch, StringComparison.OrdinalIgnoreCase) && !selectedSoftSkills.Contains(s))
                .ToList();
        }

        void FilterInterests(ChangeEventArgs e)
        {
            interestSearch = e.Value?.ToString() ?? "";
            filteredInterests = interests
                .Where(s => s.Contains(interestSearch, StringComparison.OrdinalIgnoreCase) && !selectedInterests.Contains(s))
                .ToList();
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
        #endregion

        public void Dispose()
        {
            currentDb?.Dispose();
        }
    }
}
