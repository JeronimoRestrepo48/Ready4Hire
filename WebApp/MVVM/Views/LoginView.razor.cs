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

                // Normalizar y validar email (no sanitizar con HTML encoding para JSON)
                email = email.Trim().ToLower();
                
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
                    
                    // Redireccionar directamente al chat
                    Navigation.NavigateTo("/chat", true);
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

                // Normalizar email (trim + lowercase) - no HTML encoding para JSON
                registerEmail = registerEmail.Trim().ToLower();
                
                // Sanitizar otros inputs (nombre, apellido, etc.) para prevenir XSS en la UI
                registerName = SecurityService.SanitizeInput(registerName);
                registerLastName = SecurityService.SanitizeInput(registerLastName);
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
                    
                    // Redireccionar directamente al chat
                    Navigation.NavigateTo("/chat", true);
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
            // ════════════════ TECNOLOGÍA ════════════════
            // Lenguajes de Programación
            "C#", "Java", "JavaScript", "Python", "TypeScript", "Go", "Rust", "Kotlin", "Swift", "Objective-C",
            "C++", "C", "PHP", "Ruby", "Scala", "Perl", "R", "MATLAB", "Julia", "Dart",
            "F#", "Haskell", "Elixir", "Erlang", "Clojure", "Groovy", "Lua", "VB.NET", "COBOL", "Fortran",
            
            // Frameworks Web & Mobile
            "React", "Angular", "Vue.js", "Svelte", "Next.js", "Django", "Flask", "Spring Boot", "ASP.NET Core", "Laravel",
            "React Native", "Flutter", "Ionic", "Xamarin", "Android SDK", "iOS SDK",
            
            // Bases de Datos & Cloud
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Oracle", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
            
            // Data & AI
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Data Science", "Power BI", "Tableau", "Excel Avanzado",
            
            // ════════════════ FINANZAS & CONTABILIDAD ════════════════
            "Contabilidad Financiera", "Contabilidad de Costos", "Auditoría", "NIIF", "NIC", "SAP FICO", "Oracle Financials",
            "QuickBooks", "Sage", "Teneduría de Libros", "Análisis Financiero", "Presupuestos", "Flujo de Caja",
            "Consolidación Financiera", "Impuestos", "Declaración de Renta", "Facturación Electrónica", "Tesorería",
            "Bloomberg Terminal", "Reuters Eikon", "Modelos Financieros", "Valoración de Empresas", "DCF", "VPN",
            "Gestión de Riesgos", "Derivados Financieros", "Trading", "Forex", "Análisis Técnico", "Análisis Fundamental",
            
            // ════════════════ MARKETING & PUBLICIDAD ════════════════
            "Google Ads", "Facebook Ads", "Instagram Ads", "LinkedIn Ads", "TikTok Ads", "SEO", "SEM", "Google Analytics",
            "Google Tag Manager", "HubSpot", "Mailchimp", "Salesforce Marketing Cloud", "Marketing Automation",
            "Email Marketing", "Content Marketing", "Copywriting", "Social Media Management", "Hootsuite", "Buffer",
            "Canva", "Adobe Photoshop", "Adobe Illustrator", "Adobe Premiere", "After Effects", "Final Cut Pro",
            "Community Management", "Influencer Marketing", "Growth Hacking", "A/B Testing", "CRO", "Marketing Analytics",
            
            // ════════════════ DISEÑO ════════════════
            "Adobe Creative Suite", "Photoshop", "Illustrator", "InDesign", "XD", "Sketch", "Figma", "Adobe XD",
            "CorelDRAW", "Procreate", "Blender", "3ds Max", "Maya", "Cinema 4D", "ZBrush", "AutoCAD",
            "SketchUp", "Revit", "Rhino", "SolidWorks", "CATIA", "Fusion 360", "Diseño Gráfico", "Diseño Web",
            "Diseño UX/UI", "Diseño Industrial", "Diseño de Producto", "Diseño de Moda", "Patronaje", "Corte y Confección",
            "Ilustración Digital", "Animación 2D", "Animación 3D", "Motion Graphics", "Tipografía", "Branding",
            
            // ════════════════ ARQUITECTURA & CONSTRUCCIÓN ════════════════
            "AutoCAD", "Revit", "ArchiCAD", "SketchUp", "Rhino", "3ds Max", "Lumion", "V-Ray", "Enscape",
            "BIM", "Diseño Arquitectónico", "Planos Estructurales", "Especificaciones Técnicas", "Presupuestos de Obra",
            "Gestión de Proyectos de Construcción", "Supervisión de Obra", "Control de Calidad", "Seguridad en Obra",
            "Lectura de Planos", "Cálculo Estructural", "SAP2000", "ETABS", "SAFE", "Gestión Ambiental",
            
            // ════════════════ SALUD & MEDICINA ════════════════
            "Historia Clínica Electrónica", "SOAP", "Diagnóstico Clínico", "Atención Primaria", "Farmacología",
            "Primeros Auxilios", "RCP", "ACLS", "Ecografía", "Radiología", "Laboratorio Clínico",
            "Microbiología", "Hematología", "Bioquímica Clínica", "Enfermería", "Cuidados Intensivos", "Quirófano",
            "Odontología", "Terapia Física", "Nutrición Clínica", "Psicología Clínica", "Terapia Cognitivo-Conductual",
            
            // ════════════════ DERECHO & LEGAL ════════════════
            "Derecho Civil", "Derecho Penal", "Derecho Laboral", "Derecho Comercial", "Derecho Tributario",
            "Derecho Constitucional", "Contratos", "Litigio", "Mediación", "Arbitraje", "Compliance",
            "Propiedad Intelectual", "Derecho Corporativo", "Due Diligence", "Redacción Legal", "LegalTech",
            
            // ════════════════ RECURSOS HUMANOS ════════════════
            "Reclutamiento y Selección", "Talent Acquisition", "Onboarding", "Evaluación de Desempeño",
            "Capacitación y Desarrollo", "Gestión del Talento", "Compensaciones y Beneficios", "Nómina",
            "Relaciones Laborales", "Clima Organizacional", "Cultura Organizacional", "Employer Branding",
            "HRIS", "SAP HCM", "Workday", "BambooHR", "ATS", "LinkedIn Recruiter",
            
            // ════════════════ VENTAS & COMERCIAL ════════════════
            "Ventas B2B", "Ventas B2C", "Negociación Comercial", "Prospección", "Cold Calling", "CRM",
            "Salesforce", "HubSpot CRM", "Zoho CRM", "Cierre de Ventas", "Key Account Management",
            "Gestión de Cuentas Clave", "Retail", "Merchandising", "Trade Marketing", "Atención al Cliente",
            
            // ════════════════ EDUCACIÓN ════════════════
            "Pedagogía", "Didáctica", "Planificación Curricular", "Evaluación Educativa", "Educación Especial",
            "Educación Virtual", "Moodle", "Canvas", "Google Classroom", "Zoom", "Microsoft Teams",
            "Diseño Instruccional", "Gamificación Educativa", "Metodologías Activas", "Aprendizaje Colaborativo",
            
            // ════════════════ LOGÍSTICA & SUPPLY CHAIN ════════════════
            "Gestión de Inventarios", "Planificación de Demanda", "Logística de Distribución", "Transporte",
            "Almacenamiento", "WMS", "SAP MM", "SAP SCM", "Lean Manufacturing", "Six Sigma", "Kaizen",
            "Control de Calidad", "Gestión de Proveedores", "Compras", "Importaciones", "Exportaciones",
            
            // ════════════════ GASTRONOMÍA & HOTELERÍA ════════════════
            "Cocina Profesional", "Repostería", "Pastelería", "Panadería", "Cocina Internacional",
            "Cocina Molecular", "Sommelier", "Barista", "Mixología", "Servicio al Cliente Hotelero",
            "Gestión Hotelera", "Opera PMS", "Gestión de Eventos", "Catering", "Food Cost", "HACCP",
            
            // ════════════════ ARTES & ENTRETENIMIENTO ════════════════
            "Producción Musical", "Mezcla de Audio", "Masterización", "Pro Tools", "Ableton Live", "Logic Pro",
            "FL Studio", "Composición Musical", "Teoría Musical", "Instrumentos Musicales", "Canto",
            "Actuación", "Dirección Escénica", "Fotografía", "Edición de Video", "Producción Audiovisual",
            
            // ════════════════ DEPORTES & FITNESS ════════════════
            "Entrenamiento Personal", "Nutrición Deportiva", "Kinesiología", "Preparación Física",
            "Fisioterapia Deportiva", "Coaching Deportivo", "Evaluación Física", "Planificación Deportiva",
            
            // ════════════════ CIENCIAS & INVESTIGACIÓN ════════════════
            "Metodología de Investigación", "Análisis Estadístico", "SPSS", "Stata", "SAS", "Laboratorio",
            "Técnicas de Laboratorio", "Química Analítica", "Cromatografía", "Espectroscopía", "Biotecnología",
            "Biología Molecular", "PCR", "Electroforesis", "Cultivo Celular", "Microscopía",
            
            // ════════════════ ADMINISTRACIÓN & GESTIÓN ════════════════
            "Gestión de Proyectos", "PMP", "Scrum", "Agile", "Microsoft Project", "Asana", "Trello", "Monday.com",
            "Planificación Estratégica", "Balanced Scorecard", "KPIs", "Gestión de Procesos", "BPM", "Lean", "Six Sigma",
            
            // ════════════════ IDIOMAS ════════════════
            "Inglés", "Francés", "Alemán", "Italiano", "Portugués", "Mandarín", "Japonés", "Coreano", "Árabe", "Ruso",
            "Traducción", "Interpretación", "Traducción Técnica", "Subtitulado", "Localización",
            
            // ════════════════ MEDIO AMBIENTE & ENERGÍA ════════════════
            "Energía Solar Fotovoltaica", "Energía Eólica", "Energía Hidroeléctrica", "Biomasa", "Geotermia",
            "Eficiencia Energética", "Auditoría Energética", "Certificación LEED", "BREEAM", "Huella de Carbono",
            "Análisis de Ciclo de Vida", "ISO 14001", "Gestión de Residuos", "Tratamiento de Aguas", "Remediación Ambiental",
            
            // ════════════════ AGRICULTURA & AGROINDUSTRIA ════════════════
            "Agricultura de Precisión", "Drones Agrícolas", "Sistemas de Riego", "Fertirrigación", "Hidroponía",
            "Aeroponía", "Cultivos Verticales", "Agricultura Orgánica", "BPA", "Buenas Prácticas Agrícolas",
            "Manejo Integrado de Plagas", "Postcosecha", "Agronomía", "Fitosanidad", "Suelos y Fertilización",
            
            // ════════════════ MANUFACTURA & PRODUCCIÓN ════════════════
            "TPM", "Total Productive Maintenance", "OEE", "SMED", "Kanban", "5S", "Poka-Yoke", "Value Stream Mapping",
            "Jidoka", "Heijunka", "Gemba", "Hoshin Kanri", "MRP", "ERP", "SAP PP", "Oracle Manufacturing",
            "Control Estadístico de Procesos", "SPC", "Capacidad de Procesos", "DOE", "FMEA", "APQP",
            
            // ════════════════ SEGUROS & ACTUARÍA ════════════════
            "Suscripción de Seguros", "Evaluación de Riesgos", "Actuaría", "Seguros de Vida", "Seguros Generales",
            "Reaseguros", "Claims Management", "Seguros de Salud", "Pensiones", "Análisis Actuarial",
            
            // ════════════════ INMOBILIARIA & BIENES RAÍCES ════════════════
            "Valoración Inmobiliaria", "Tasación", "Corretaje Inmobiliario", "Gestión de Propiedades", "Property Management",
            "Desarrollo Inmobiliario", "Inversión Inmobiliaria", "REITs", "Due Diligence Inmobiliario",
            
            // ════════════════ PERIODISMO & MEDIOS ════════════════
            "Redacción Periodística", "Edición de Contenidos", "Fact-Checking", "Investigación Periodística",
            "Periodismo de Datos", "Periodismo Multimedia", "Producción de Noticias", "Locución", "Presentación TV",
            "Guionismo", "Edición de Audio", "Postproducción", "Color Grading", "VFX", "After Effects",
            
            // ════════════════ RELACIONES INTERNACIONALES ════════════════
            "Diplomacia", "Cooperación Internacional", "Derecho Internacional Público", "Comercio Exterior",
            "Aduanas", "Incoterms", "Negociación Internacional", "Geopolítica", "Análisis Político",
            
            // ════════════════ MINERÍA & PETRÓLEO ════════════════
            "Geología", "Exploración Minera", "Perforación", "Voladura", "Operaciones Mineras",
            "Procesamiento de Minerales", "Metalurgia", "Refinación", "Petróleo y Gas", "Upstream", "Downstream",
            
            // ════════════════ TEXTIL & MODA ════════════════
            "Diseño de Moda", "Patronaje Industrial", "Confección", "Textiles", "Fibras Textiles",
            "Producción Textil", "Control de Calidad Textil", "Tendencias de Moda", "Fashion Styling",
            "Visual Merchandising", "Fashion Buying", "Forecasting de Moda",
            
            // ════════════════ TELECOMUNICACIONES ════════════════
            "Redes de Telecomunicaciones", "5G", "LTE", "Fibra Óptica", "Redes Inalámbricas", "RF",
            "Transmisión", "Switching", "VoIP", "SIP", "Telefonía IP", "NOC", "Network Operations Center",
            
            // ════════════════ AVIACIÓN & AEROESPACIAL ════════════════
            "Mantenimiento Aeronáutico", "Mecánica de Vuelo", "Navegación Aérea", "Control de Tráfico Aéreo",
            "Operaciones Aeroportuarias", "Ground Handling", "Aviación Comercial", "Seguridad Aeroportuaria",
            
            // ════════════════ MARÍTIMO & PORTUARIO ════════════════
            "Operaciones Portuarias", "Logística Marítima", "Derecho Marítimo", "Navegación Marítima",
            "Gestión Naviera", "Ship Management", "Charter", "Transporte Marítimo",
            
            // ════════════════ FARMACÉUTICA & QUÍMICA ════════════════
            "Formulación Farmacéutica", "GMP", "Validación de Procesos", "Regulatory Affairs", "Farmacovigilancia",
            "Ensayos Clínicos Fase I-IV", "QA Farmacéutica", "QC Farmacéutica", "Síntesis Orgánica",
            
            // ════════════════ VIDEOJUEGOS & GAMING ════════════════
            "Game Design", "Level Design", "Game Programming", "Unity 3D", "Unreal Engine 5", "Godot",
            "C++ para Videojuegos", "Shaders", "Rendering", "Rigging", "Animación de Personajes", "Game Art",
            "Sound Design", "Game Testing", "QA Gaming", "Monetización de Juegos", "LiveOps",
            
            // ════════════════ CINE & PRODUCCIÓN ════════════════
            "Dirección de Fotografía", "Cinematografía", "Iluminación Cinematográfica", "Grip", "Gaffer",
            "Script", "Continuidad", "Producción Ejecutiva", "Dirección de Arte", "Escenografía",
            "Diseño de Vestuario", "Maquillaje Cinematográfico", "Efectos Especiales", "Storyboard",
            
            // ════════════════ PUBLICIDAD & RELACIONES PÚBLICAS ════════════════
            "Planificación de Medios", "Compra de Medios", "Estrategia Publicitaria", "Dirección Creativa",
            "Redacción Publicitaria", "Producción Publicitaria", "Media Planning", "Media Buying",
            "Crisis Communication", "Relaciones con Medios", "Press Release", "Event Planning",
            
            // ════════════════ INVESTIGACIÓN DE MERCADOS ════════════════
            "Estudios de Mercado", "Focus Groups", "Encuestas", "Análisis de Consumidor", "Shopper Insights",
            "Brand Tracking", "Segmentación de Mercados", "Pricing Research", "Neuromarketing",
            
            // ════════════════ CALIDAD & NORMALIZACIÓN ════════════════
            "ISO 9001", "ISO 14001", "ISO 45001", "OHSAS 18001", "ISO 27001", "ISO 22000", "HACCP",
            "BRC", "IFS", "Auditoría Interna", "Auditoría de Calidad", "Control de Documentos",
            "No Conformidades", "Acciones Correctivas", "Mejora Continua", "Kaizen"
        };
        private List<string> softSkills = new()
        {
            // Comunicación
            "Comunicación efectiva", "Comunicación escrita", "Comunicación verbal", "Presentaciones públicas", "Oratoria",
            "Storytelling", "Comunicación intercultural", "Escucha activa", "Feedback constructivo", "Claridad expositiva",
            "Comunicación persuasiva", "Negociación", "Mediación", "Expresión corporal", "Comunicación digital",
            
            // Liderazgo
            "Liderazgo", "Liderazgo transformacional", "Liderazgo situacional", "Gestión de equipos", "Desarrollo de talento",
            "Mentoría", "Coaching", "Delegación efectiva", "Empoderamiento", "Gestión del cambio", "Visión estratégica",
            "Inspiración", "Toma de decisiones", "Autoridad natural", "Liderazgo remoto",
            
            // Trabajo en Equipo
            "Trabajo en equipo", "Colaboración", "Cooperación", "Trabajo remoto", "Trabajo híbrido",
            "Coordinación", "Sinergia", "Construcción de consenso", "Trabajo interdisciplinario", "Networking",
            
            // Pensamiento
            "Pensamiento crítico", "Pensamiento analítico", "Pensamiento estratégico", "Pensamiento creativo",
            "Pensamiento sistémico", "Pensamiento lateral", "Resolución de problemas", "Análisis de datos",
            "Síntesis de información", "Razonamiento lógico", "Toma de decisiones complejas",
            
            // Inteligencia Emocional
            "Inteligencia emocional", "Empatía", "Autoconciencia", "Autorregulación", "Gestión emocional",
            "Comprensión emocional", "Sensibilidad interpersonal", "Asertividad", "Gestión del estrés",
            "Resiliencia", "Equilibrio emocional", "Compasión", "Paciencia",
            
            // Adaptabilidad
            "Adaptabilidad", "Flexibilidad", "Agilidad mental", "Versatilidad", "Apertura al cambio",
            "Tolerancia a la ambigüedad", "Gestión de la incertidumbre", "Innovación", "Experimentación",
            "Aprendizaje continuo", "Curiosidad", "Mentalidad de crecimiento", "Actualización constante",
            
            // Gestión Personal
            "Gestión del tiempo", "Organización", "Planificación", "Priorización", "Productividad",
            "Autodisciplina", "Autogestión", "Puntualidad", "Cumplimiento de plazos", "Multitasking",
            "Gestión de prioridades", "Enfoque", "Concentración", "Eficiencia",
            
            // Creatividad e Innovación
            "Creatividad", "Innovación", "Pensamiento disruptivo", "Ideación", "Brainstorming",
            "Soluciones creativas", "Originalidad", "Imaginación", "Visión innovadora",
            
            // Orientación a Resultados
            "Orientación a resultados", "Orientación al logro", "Proactividad", "Iniciativa",
            "Determinación", "Perseverancia", "Compromiso", "Responsabilidad", "Accountability",
            "Orientación al detalle", "Calidad", "Excelencia", "Superación de metas",
            
            // Cliente y Servicio
            "Orientación al cliente", "Servicio al cliente", "Satisfacción del cliente", "Gestión de expectativas",
            "Atención personalizada", "Resolución de quejas", "Experiencia de usuario",
            
            // Gestión de Conflictos
            "Gestión de conflictos", "Resolución de disputas", "Diplomacia", "Tacto", "Manejo de crisis",
            "Negociación win-win", "Mediación de conflictos",
            
            // Ética y Valores
            "Ética profesional", "Integridad", "Honestidad", "Transparencia", "Confidencialidad",
            "Respeto", "Inclusión", "Diversidad", "Equidad", "Justicia", "Responsabilidad social",
            
            // Otros
            "Cultura de feedback", "Motivación", "Entusiasmo", "Optimismo", "Energía positiva",
            "Influencia", "Persuasión", "Capacidad de síntesis", "Visión holística", "Gestión de proyectos",
            
            // ════════════════ HABILIDADES INTERPERSONALES AVANZADAS ════════════════
            "Construcción de relaciones", "Rapport", "Carisma", "Presencia ejecutiva", "Gravitas",
            "Conexión emocional", "Empatía cultural", "Sensibilidad social", "Lectura de lenguaje corporal",
            "Inteligencia social", "Habilidades políticas", "Networking estratégico", "Construcción de alianzas",
            
            // ════════════════ HABILIDADES DE COMUNICACIÓN AVANZADAS ════════════════
            "Comunicación asertiva", "Comunicación no violenta", "Comunicación ejecutiva", "Executive presence",
            "Storytelling empresarial", "Presentaciones de alto impacto", "TED-style presentations",
            "Comunicación de crisis", "Manejo de medios", "Media training", "Portavocía", "Vocería corporativa",
            
            // ════════════════ LIDERAZGO AVANZADO ════════════════
            "Liderazgo adaptativo", "Liderazgo ágil", "Liderazgo de influencia", "Servant leadership",
            "Liderazgo auténtico", "Liderazgo distribuido", "Liderazgo digital", "Gestión remota de equipos",
            "Liderazgo multicultural", "Liderazgo generacional", "Desarrollo de líderes", "Sucesión de liderazgo",
            
            // ════════════════ PENSAMIENTO ESTRATÉGICO Y ANALÍTICO ════════════════
            "Visión estratégica", "Pensamiento de sistemas", "Modelado de escenarios", "Previsión estratégica",
            "Análisis FODA", "Análisis PESTEL", "Cinco Fuerzas de Porter", "Canvas de modelo de negocio",
            "Design Thinking avanzado", "Pensamiento de diseño", "Solución creativa de problemas",
            
            // ════════════════ GESTIÓN Y ORGANIZACIÓN ════════════════
            "Priorización efectiva", "Matriz de Eisenhower", "Método GTD", "Método Pomodoro",
            "Time blocking", "Deep work", "Gestión de energía", "Ritmos ultradianos",
            "Gestión de múltiples proyectos", "Delegación estratégica", "Follow-up efectivo",
            
            // ════════════════ INNOVACIÓN Y CREATIVIDAD ════════════════
            "Pensamiento divergente", "Pensamiento convergente", "SCAMPER", "Técnicas de ideación",
            "Prototipado rápido", "Iteración", "Experimentación", "Tolerancia al fracaso",
            "Mentalidad de principiante", "Beginner's mind", "Cuestionamiento de supuestos",
            
            // ════════════════ AGILIDAD Y ADAPTABILIDAD ════════════════
            "Agilidad de aprendizaje", "Learning agility", "Agilidad emocional", "Comfort con la ambigüedad",
            "Adaptación al cambio", "Pivoteo", "Reframing", "Cambio de perspectiva",
            "Desaprendizaje", "Unlearning", "Actualización continua", "Lifelong learning",
            
            // ════════════════ INTELIGENCIA EMOCIONAL AVANZADA ════════════════
            "Autogestión emocional", "Regulación emocional", "Conciencia emocional avanzada",
            "Empatía cognitiva", "Empatía afectiva", "Empatía compasiva", "Manejo de emociones difíciles",
            "Ventana de tolerancia", "Co-regulación", "Presencia mindful",
            
            // ════════════════ NEGOCIACIÓN Y PERSUASIÓN AVANZADAS ════════════════
            "Negociación integrativa", "BATNA", "ZOPA", "Anclaje en negociación", "Concesiones estratégicas",
            "Negociación multicultural", "Persuasión ética", "Influencia sin autoridad", "Arquitectura de elección",
            
            // ════════════════ COLABORACIÓN Y TRABAJO EN EQUIPO ════════════════
            "Facilitación de grupos", "Dinámicas de grupo", "Resolución de conflictos grupales",
            "Construcción de consenso", "Toma de decisiones grupales", "Técnica Delphi", "Brainwriting",
            "Trabajo asíncrono", "Colaboración virtual", "Gestión de equipos distribuidos",
            
            // ════════════════ RESILIENCIA Y BIENESTAR ════════════════
            "Gestión del burnout", "Prevención de agotamiento", "Autocuidado", "Establecimiento de límites",
            "Work-life integration", "Psicología positiva", "Fortalezas de carácter", "Gratitud",
            "Mindfulness", "Meditación", "Respiración consciente", "Técnicas de relajación",
            
            // ════════════════ HABILIDADES DIGITALES Y TECNOLÓGICAS ════════════════
            "Alfabetización digital", "Ciudadanía digital", "Seguridad en línea", "Privacidad digital",
            "Trabajo remoto efectivo", "Colaboración digital", "Netiqueta", "Comunicación virtual",
            "Gestión de identidad digital", "Marca personal digital", "Pensamiento computacional"
        };
        private List<string> interests = new()
        {
            // ════════════════ TECNOLOGÍA ════════════════
            "Inteligencia Artificial", "Machine Learning", "Deep Learning", "Computer Vision", "NLP",
            "Desarrollo Web", "Desarrollo Mobile", "Cloud Computing", "Ciberseguridad", "Blockchain",
            "Data Science", "Big Data", "DevOps", "IoT", "Robótica", "eSports", "Videojuegos",
            
            // ════════════════ FINANZAS & NEGOCIOS ════════════════
            "Finanzas Personales", "Inversiones", "Mercado de Valores", "Trading", "Criptomonedas",
            "Análisis Financiero", "Banca", "Seguros", "Bienes Raíces", "Fondos de Inversión",
            "Private Equity", "Venture Capital", "Finanzas Corporativas", "Contabilidad", "Auditoría",
            "Planificación Financiera", "Gestión de Riesgos", "Forex", "Commodities", "Derivados Financieros",
            "FinTech", "Banca Digital", "Pagos Digitales", "Open Banking", "RegTech", "InsurTech",
            "Microfinanzas", "Crowdfunding", "Economía", "Macroeconomía", "Microeconomía",
            
            // ════════════════ MARKETING & COMUNICACIÓN ════════════════
            "Marketing Digital", "Social Media", "Content Marketing", "Email Marketing", "SEO", "SEM",
            "Publicidad Digital", "Branding", "Storytelling", "Copywriting", "Influencer Marketing",
            "Marketing de Contenidos", "Inbound Marketing", "Growth Marketing", "Performance Marketing",
            "Marketing Analytics", "CRM", "Marketing Automation", "Community Management",
            "Relaciones Públicas", "Comunicación Corporativa", "Comunicación Estratégica",
            "Periodismo", "Periodismo Digital", "Producción Audiovisual", "Fotografía", "Video Marketing",
            
            // ════════════════ DISEÑO & CREATIVIDAD ════════════════
            "Diseño Gráfico", "Diseño Web", "Diseño UX/UI", "Diseño Industrial", "Diseño de Producto",
            "Diseño de Moda", "Diseño de Interiores", "Arquitectura", "Paisajismo", "Urbanismo",
            "Ilustración", "Animación", "Motion Graphics", "Arte Digital", "Bellas Artes",
            "Fotografía Artística", "Fotografía Comercial", "Fotografía de Moda", "Retoque Fotográfico",
            "Tipografía", "Packaging", "Editorial Design", "Brand Identity", "Design Thinking",
            
            // ════════════════ ARQUITECTURA & CONSTRUCCIÓN ════════════════
            "Arquitectura Sostenible", "Arquitectura Bioclimática", "BIM", "Diseño Arquitectónico",
            "Construcción", "Gestión de Proyectos de Construcción", "Ingeniería Civil",
            "Ingeniería Estructural", "Diseño Urbano", "Patrimonio Arquitectónico", "Restauración",
            "Eficiencia Energética", "Certificaciones LEED", "Smart Buildings",
            
            // ════════════════ SALUD & BIENESTAR ════════════════
            "Medicina", "Enfermería", "Farmacia", "Nutrición", "Dietética", "Salud Pública",
            "Epidemiología", "Medicina Preventiva", "Telemedicina", "HealthTech", "Biotecnología",
            "Investigación Médica", "Ensayos Clínicos", "Farmacología", "Genética", "Biología Molecular",
            "Psicología", "Psicología Clínica", "Psicoterapia", "Neurociencia", "Salud Mental",
            "Terapia Cognitivo-Conductual", "Mindfulness", "Meditación", "Yoga", "Bienestar",
            "Fitness", "Entrenamiento Personal", "Nutrición Deportiva", "Rehabilitación",
            "Fisioterapia", "Kinesiología", "Terapia Ocupacional", "Medicina Deportiva",
            
            // ════════════════ EDUCACIÓN ════════════════
            "Educación", "Pedagogía", "Didáctica", "Educación Virtual", "E-Learning", "EdTech",
            "Innovación Educativa", "Metodologías Activas", "Gamificación Educativa",
            "Educación Especial", "Educación Inicial", "Educación Primaria", "Educación Secundaria",
            "Educación Superior", "Formación Profesional", "Capacitación Corporativa",
            "Diseño Instruccional", "Currículo", "Evaluación Educativa", "Investigación Educativa",
            
            // ════════════════ DERECHO & LEGAL ════════════════
            "Derecho", "Derecho Civil", "Derecho Penal", "Derecho Laboral", "Derecho Comercial",
            "Derecho Tributario", "Derecho Constitucional", "Derecho Internacional", "Derechos Humanos",
            "Propiedad Intelectual", "Derecho Digital", "Derecho Ambiental", "Derecho de Familia",
            "Litigio", "Arbitraje", "Mediación", "Compliance", "LegalTech", "Justicia",
            
            // ════════════════ RECURSOS HUMANOS & TALENTO ════════════════
            "Gestión del Talento", "Reclutamiento", "Selección de Personal", "Capacitación",
            "Desarrollo Organizacional", "Cultura Organizacional", "Clima Laboral",
            "Compensaciones", "Employee Experience", "Employer Branding", "People Analytics",
            "Diversidad e Inclusión", "Bienestar Laboral", "Liderazgo Organizacional",
            
            // ════════════════ VENTAS & COMERCIO ════════════════
            "Ventas", "Negociación", "Comercio Internacional", "E-commerce", "Retail",
            "Merchandising", "Trade Marketing", "Key Account Management", "Business Development",
            "Atención al Cliente", "Customer Experience", "Customer Success", "Servicio al Cliente",
            
            // ════════════════ LOGÍSTICA & SUPPLY CHAIN ════════════════
            "Logística", "Supply Chain", "Gestión de Inventarios", "Almacenamiento", "Transporte",
            "Distribución", "Compras", "Gestión de Proveedores", "Comercio Exterior",
            "Importaciones", "Exportaciones", "Aduanas", "Lean Manufacturing", "Six Sigma",
            
            // ════════════════ GASTRONOMÍA & HOTELERÍA ════════════════
            "Gastronomía", "Cocina", "Repostería", "Panadería", "Cocina Internacional",
            "Alta Cocina", "Cocina Fusión", "Cocina Saludable", "Cocina Vegana", "Sommelier",
            "Enología", "Cerveza Artesanal", "Barismo", "Mixología", "Hotelería", "Turismo",
            "Gestión Hotelera", "Gestión de Restaurantes", "Catering", "Eventos",
            
            // ════════════════ ARTES & ENTRETENIMIENTO ════════════════
            "Música", "Producción Musical", "Composición", "Instrumentos Musicales", "Canto",
            "Teatro", "Actuación", "Dirección Escénica", "Dramaturgia", "Danza", "Coreografía",
            "Cine", "Dirección Cinematográfica", "Guión", "Edición de Video", "Sonido",
            "Artes Plásticas", "Pintura", "Escultura", "Arte Contemporáneo", "Historia del Arte",
            "Literatura", "Escritura Creativa", "Poesía", "Novela", "Crítica Literaria",
            
            // ════════════════ DEPORTES ════════════════
            "Fútbol", "Basketball", "Tenis", "Atletismo", "Natación", "Ciclismo", "Running",
            "CrossFit", "Artes Marciales", "Boxeo", "MMA", "Entrenamiento Deportivo",
            "Coaching Deportivo", "Gestión Deportiva", "Marketing Deportivo", "Periodismo Deportivo",
            
            // ════════════════ CIENCIAS & INVESTIGACIÓN ════════════════
            "Investigación Científica", "Biología", "Química", "Física", "Matemáticas",
            "Estadística", "Biotecnología", "Biología Molecular", "Genética", "Microbiología",
            "Química Analítica", "Química Orgánica", "Bioquímica", "Astronomía", "Astrofísica",
            "Geología", "Ciencias Ambientales", "Ecología", "Conservación", "Cambio Climático",
            
            // ════════════════ INGENIERÍA (NO SOFTWARE) ════════════════
            "Ingeniería Industrial", "Ingeniería Mecánica", "Ingeniería Eléctrica",
            "Ingeniería Electrónica", "Ingeniería Química", "Ingeniería Ambiental",
            "Ingeniería de Procesos", "Automatización Industrial", "Manufactura",
            "Control de Calidad", "Mejora Continua", "Gestión de Operaciones",
            
            // ════════════════ SOSTENIBILIDAD & MEDIO AMBIENTE ════════════════
            "Sostenibilidad", "Energías Renovables", "Energía Solar", "Energía Eólica",
            "Eficiencia Energética", "Economía Circular", "Reciclaje", "Gestión Ambiental",
            "Desarrollo Sostenible", "ESG", "Responsabilidad Social", "GreenTech",
            "Agricultura Sostenible", "Permacultura", "AgriTech",
            
            // ════════════════ EMPRENDIMIENTO & STARTUPS ════════════════
            "Emprendimiento", "Startups", "Innovación", "Business Model Canvas", "Lean Startup",
            "Product Management", "Growth Hacking", "Venture Capital", "Angel Investing",
            "Pitch", "Fundraising", "Ecosistema Emprendedor", "Aceleradoras", "Incubadoras",
            
            // ════════════════ ADMINISTRACIÓN & GESTIÓN ════════════════
            "Administración de Empresas", "Gestión de Proyectos", "Planificación Estratégica",
            "Gestión de Procesos", "Gestión del Cambio", "Consultoría", "Consultoría Estratégica",
            "Análisis de Negocios", "Inteligencia de Negocios", "Transformación Digital",
            
            // ════════════════ IDIOMAS & CULTURAS ════════════════
            "Aprendizaje de Idiomas", "Inglés", "Francés", "Alemán", "Italiano", "Portugués",
            "Mandarín", "Japonés", "Coreano", "Árabe", "Traducción", "Interpretación",
            "Culturas del Mundo", "Estudios Interculturales", "Antropología",
            
            // ════════════════ DESARROLLO PERSONAL ════════════════
            "Desarrollo Personal", "Productividad", "Gestión del Tiempo", "Hábitos",
            "Motivación", "Liderazgo Personal", "Coaching", "Mentoring", "Oratoria",
            "Hablar en Público", "Networking", "Marca Personal", "Career Development",
            
            // ════════════════ MEDIO AMBIENTE & SOSTENIBILIDAD ════════════════
            "Energía Solar", "Energía Eólica", "Energía Hidroeléctrica", "Biomasa", "Hidrógeno Verde",
            "Economía Verde", "Finanzas Sostenibles", "Inversión ESG", "Bonos Verdes",
            "Cambio Climático", "Mitigación", "Adaptación Climática", "Huella de Carbono",
            "Neutralidad de Carbono", "Net Zero", "Conservación Ambiental", "Biodiversidad",
            "Gestión de Residuos", "Economía del Reciclaje", "Upcycling", "Zero Waste",
            
            // ════════════════ AGRICULTURA & ALIMENTACIÓN ════════════════
            "Agricultura Sostenible", "Permacultura", "Agroecología", "Agricultura Orgánica",
            "Seguridad Alimentaria", "Sistemas Alimentarios", "Food Tech", "Agricultura Vertical",
            "Urban Farming", "Acuaponía", "Insect Farming", "Carne Cultivada", "Proteínas Alternativas",
            
            // ════════════════ MANUFACTURA & INDUSTRIA 4.0 ════════════════
            "Industria 4.0", "Smart Manufacturing", "Digital Twin", "IIoT", "Manufactura Aditiva",
            "Impresión 3D Industrial", "Cobots", "Robótica Colaborativa", "AGVs", "AMRs",
            "Realidad Aumentada Industrial", "Mantenimiento Predictivo", "Edge Computing Industrial",
            
            // ════════════════ SEGUROS & GESTIÓN DE RIESGOS ════════════════
            "Gestión de Riesgos Empresariales", "ERM", "Riesgo Operacional", "Riesgo Financiero",
            "Riesgo de Crédito", "Riesgo de Mercado", "Riesgo Reputacional", "Seguros",
            "InsurTech", "Actuaría", "Modelado de Riesgos", "Análisis Cuantitativo de Riesgos",
            
            // ════════════════ INMOBILIARIO & CONSTRUCCIÓN ════════════════
            "Real Estate", "Desarrollo Inmobiliario", "PropTech", "Smart Buildings",
            "Edificios Inteligentes", "Domótica", "Automatización de Edificios", "BMS",
            "Construcción Sostenible", "Materiales Sostenibles", "Construcción Modular",
            "Prefabricación", "Construcción Industrializada",
            
            // ════════════════ RETAIL & E-COMMERCE ════════════════
            "Retail Innovation", "Omnichannel", "Experiencia Omnicanal", "Click & Collect",
            "Last Mile Delivery", "Fulfillment", "Amazon FBA", "Dropshipping", "Marketplace Management",
            "Retail Analytics", "Merchandising Visual", "Category Management", "Private Label",
            
            // ════════════════ MEDIOS & ENTRETENIMIENTO ════════════════
            "Streaming", "OTT", "Content Strategy", "Producción de Contenidos", "Media Planning",
            "Programática", "Publicidad Programática", "AdTech", "MarTech", "Podcast Production",
            "Video Streaming", "Live Streaming", "Transmedia Storytelling",
            
            // ════════════════ TELECOMUNICACIONES & CONECTIVIDAD ════════════════
            "5G", "6G", "Redes Móviles", "Conectividad", "Banda Ancha", "Fibra Óptica",
            "Redes Inalámbricas", "WiFi 6", "SD-WAN", "Network Automation", "NFV", "SDN",
            
            // ════════════════ AVIACIÓN & MOVILIDAD AÉREA ════════════════
            "Aviación", "Aviación Comercial", "Aviación Ejecutiva", "Drones Comerciales",
            "Urban Air Mobility", "eVTOL", "Movilidad Aérea Urbana", "Gestión Aeroportuaria",
            
            // ════════════════ AUTOMOTRIZ & MOVILIDAD ════════════════
            "Movilidad Urbana", "Vehículos Eléctricos", "EV", "Vehículos Autónomos", "ADAS",
            "Movilidad como Servicio", "MaaS", "Car Sharing", "Ride Hailing", "Micromobilidad",
            "Scooters Eléctricos", "Bicicletas Eléctricas", "Movilidad Sostenible",
            
            // ════════════════ FARMACÉUTICA & BIOTECH ════════════════
            "Farmacéutica", "Biotecnología", "Terapias Génicas", "Medicina Personalizada",
            "Medicina de Precisión", "Inmunoterapia", "Vacunas", "Desarrollo de Fármacos",
            "Drug Discovery", "Bioinformática", "Biología Sintética", "CRISPR",
            
            // ════════════════ VIDEOJUEGOS & GAMING ════════════════
            "Gaming", "Game Development", "Game Design", "Esports Management", "Game Streaming",
            "Cloud Gaming", "Mobile Gaming", "Casual Games", "Hyper-Casual Games", "Indie Games",
            "Game Monetization", "In-App Purchases", "Game Analytics", "Player Retention",
            
            // ════════════════ CINE & AUDIOVISUAL ════════════════
            "Producción Cinematográfica", "Post-producción", "VFX", "CGI", "Animación CGI",
            "Dirección de Cine", "Cinematografía Digital", "4K", "8K", "HDR", "Dolby Atmos",
            
            // ════════════════ PUBLICIDAD & BRANDING ════════════════
            "Brand Management", "Brand Strategy", "Posicionamiento de Marca", "Arquitectura de Marca",
            "Rebranding", "Brand Experience", "Experiential Marketing", "Activaciones de Marca",
            
            // ════════════════ ANALÍTICA & BIG DATA ════════════════
            "Business Analytics", "Predictive Analytics", "Prescriptive Analytics", "Data Mining",
            "Data Warehouse", "Data Lake", "Data Mesh", "Data Fabric", "DataOps",
            "Analytics Engineering", "Reverse ETL", "Customer Data Platform", "CDP",
            
            // ════════════════ CADENA DE SUMINISTRO ════════════════
            "Supply Chain 4.0", "Digital Supply Chain", "Supply Chain Analytics",
            "Demand Forecasting", "Inventory Optimization", "Supply Chain Visibility",
            "Last Mile Logistics", "Warehouse Automation", "Robotics in Logistics",
            
            // ════════════════ CALIDAD & EXCELENCIA OPERACIONAL ════════════════
            "Excelencia Operacional", "Operational Excellence", "Total Quality Management",
            "Continuous Improvement", "Process Excellence", "Benchmarking", "Best Practices",
            
            // ════════════════ RECURSOS NATURALES & MINERÍA ════════════════
            "Minería", "Minería Sostenible", "Exploración Mineral", "Geología Aplicada",
            "Petróleo y Gas", "Oil & Gas", "Energía Convencional", "Upstream Oil & Gas",
            
            // ════════════════ TURISMO & HOSPITALIDAD ════════════════
            "Turismo", "Gestión Turística", "Turismo Sostenible", "Ecoturismo", "Turismo Cultural",
            "Turismo de Aventura", "Gestión de Destinos", "Revenue Management", "Yield Management",
            "Hospitalidad", "Gestión de Experiencias", "Customer Journey Mapping",
            
            // ════════════════ GOBIERNO & SECTOR PÚBLICO ════════════════
            "Políticas Públicas", "Gestión Pública", "Gobierno Digital", "E-Government",
            "Smart Cities", "GovTech", "Participación Ciudadana", "Transparencia", "Open Data",
            
            // ════════════════ ONGs & DESARROLLO SOCIAL ════════════════
            "Desarrollo Social", "Impacto Social", "Empresas B", "B Corps", "Economía Social",
            "Emprendimiento Social", "Social Innovation", "Filantropía", "Fundraising",
            "Gestión de ONGs", "Voluntariado", "Cooperación al Desarrollo",
            
            // ════════════════ INVESTIGACIÓN & ACADEMIA ════════════════
            "Investigación Académica", "Metodología Científica", "Publicación Científica",
            "Revisión por Pares", "Gestión de Investigación", "Transferencia Tecnológica",
            "Divulgación Científica", "Science Communication",
            
            // ════════════════ PROPIEDAD INTELECTUAL ════════════════
            "Patentes", "Marcas", "Derechos de Autor", "Copyright", "Licenciamiento",
            "Transferencia de Tecnología", "IP Management", "IP Strategy",
            
            // ════════════════ FUTURO DEL TRABAJO ════════════════
            "Future of Work", "Trabajo Híbrido", "Distributed Teams", "Async Work",
            "Remote-First", "Digital Nomadism", "Gig Economy", "Freelancing",
            "Portfolio Careers", "Upskilling", "Reskilling", "Lifelong Learning"
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
                filteredHardSkills.Clear();
            }
            else
            {
                // Búsqueda mejorada: coincidencia al inicio tiene prioridad, luego contenido
                var searchLower = hardSkillSearch.ToLower();
                filteredHardSkills = hardSkills
                    .Where(s => !selectedHardSkills.Contains(s) && 
                               s.ToLower().Contains(searchLower))
                    .OrderBy(s => !s.ToLower().StartsWith(searchLower)) // Priorizar coincidencias al inicio
                    .ThenBy(s => s.Length) // Luego por longitud (más cortos primero)
                    .ThenBy(s => s)
                    .Take(15) // Aumentado a 15 resultados
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
                filteredSoftSkills.Clear();
            }
            else
            {
                // Búsqueda mejorada con priorización
                var searchLower = softSkillSearch.ToLower();
                filteredSoftSkills = softSkills
                    .Where(s => !selectedSoftSkills.Contains(s) && 
                               s.ToLower().Contains(searchLower))
                    .OrderBy(s => !s.ToLower().StartsWith(searchLower))
                    .ThenBy(s => s.Length)
                    .ThenBy(s => s)
                    .Take(15)
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
                filteredInterests.Clear();
            }
            else
            {
                // Búsqueda mejorada con priorización
                var searchLower = interestSearch.ToLower();
                filteredInterests = interests
                    .Where(i => !selectedInterests.Contains(i) && 
                               i.ToLower().Contains(searchLower))
                    .OrderBy(i => !i.ToLower().StartsWith(searchLower))
                    .ThenBy(i => i.Length)
                    .ThenBy(i => i)
                    .Take(15)
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
