using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Ready4Hire.MVVM.Views
{
    public partial class GamificationView : ComponentBase
    {
        [Inject]
        public GamificationService GamificationService { get; set; } = null!;
        
        [Inject]
        public NavigationManager Navigation { get; set; } = null!;

        [Inject]
        public AchievementProgressService AchievementProgressService { get; set; } = null!;

        [Inject]
        public IDbContextFactory<AppDbContext> DbFactory { get; set; } = null!;

        private UserStats? stats;
        private List<Achievement> achievements = new();
        private List<GameInfo> games = new();
        private List<LeaderboardEntry> leaderboard = new();
        
        private string userId = "1"; // TODO: Get from auth service
        private string activeTab = "stats";
        private bool isLoading = true;
        private bool chartsInitialized = false;

        protected override async Task OnInitializedAsync()
        {
            await LoadData();
        }

        private async Task LoadData()
        {
            isLoading = true;
            StateHasChanged();

            try
            {
                // Por ahora usar datos mock directamente para mostrar información variada
                games = GetMockGames();
                
                // Intentar cargar datos del backend en paralelo
                var statsTask = GamificationService.GetUserStatsAsync(userId);
                var achievementsTask = GamificationService.GetAchievementsAsync(userId);
                var leaderboardTask = GamificationService.GetLeaderboardAsync(limit: 50);

                // Si el backend responde, usar esos datos (excepto games que ya están actualizados)
                try 
                {
                    await Task.WhenAll(statsTask, achievementsTask, leaderboardTask);
                    stats = await statsTask;
                    achievements = await achievementsTask ?? new();
                    leaderboard = await leaderboardTask ?? new();
                    
                    // Enriquecer el leaderboard con nombres reales de usuarios desde la base de datos
                    await EnrichLeaderboardWithRealNames(leaderboard);
                }
                catch (Exception backendEx)
                {
                    Console.WriteLine($"Backend not available, using mock data: {backendEx.Message}");
                    // Usar stats mock también
                    stats = GetMockStats();
                    achievements = await AchievementProgressService.GetUserAchievementsAsync(userId);
                    leaderboard = GetMockLeaderboard();
                    
                    // Enriquecer el leaderboard mock con nombres reales si es posible
                    await EnrichLeaderboardWithRealNames(leaderboard);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading gamification data: {ex.Message}");
                // Usar datos mock como fallback completo
                games = GetMockGames();
                stats = GetMockStats();
                achievements = await AchievementProgressService.GetUserAchievementsAsync(userId);
                leaderboard = GetMockLeaderboard();
                
                // Enriquecer el leaderboard mock con nombres reales si es posible
                await EnrichLeaderboardWithRealNames(leaderboard);
            }
            finally
            {
                isLoading = false;
                StateHasChanged();
            }
        }

        private double GetXpPercentage()
        {
            if (stats == null) return 0;
            int nextLevelXp = GetNextLevelXp();
            int currentLevelXp = stats.Level * 1000;
            int xpInCurrentLevel = stats.Experience - currentLevelXp;
            return (double)xpInCurrentLevel / 1000 * 100;
        }

        private int GetNextLevelXp()
        {
            if (stats == null) return 1000;
            return (stats.Level + 1) * 1000;
        }

        private string GetGameIcon(string gameType)
        {
            return gameType.ToLower() switch
            {
                // 5 juegos originales
                "code_challenge" => "💻",
                "quick_quiz" => "⚡",
                "scenario_simulator" => "🎯", 
                "speed_round" => "🕐",
                "skill_builder" => "🛠️",
                
                // 10 juegos universales nuevos
                "memory_challenge" => "🧠",
                "logic_puzzles" => "🧩",
                "time_management" => "⏰",
                "communication_skills" => "💬",
                "problem_solving_race" => "🏃",
                "decision_trees" => "🌳",
                "pattern_recognition" => "👁️",
                "critical_thinking" => "🤔",
                "innovation_lab" => "💡",
                "stress_test" => "😰",
                
                // Íconos adicionales para otros juegos
                "problem_solver" => "🧩",
                "debugging_challenge" => "🐛",
                "system_design" => "🏗️",
                "algorithm_race" => "🏁",
                "code_review" => "👀",
                "api_builder" => "🔌",
                "database_quest" => "🗄️",
                "security_audit" => "🔐",
                "performance_optimizer" => "⚡",
                _ => "🎮"
            };
        }

        private string GetRankMedal(int rank)
        {
            return rank switch
            {
                1 => "🥇",
                2 => "🥈",
                3 => "🥉",
                _ => $"#{rank}"
            };
        }

        private async Task StartGame(string gameId, string gameType)
        {
            Console.WriteLine($"Starting game: {gameId} ({gameType})");
            
            // Actualizar progreso de logros cuando se inicia un juego
            await AchievementProgressService.UpdateAchievementsForActionAsync(userId, "game_completed");
            
            // Todos los 15 juegos disponibles
            var route = gameType.ToLower() switch
            {
                // 5 juegos originales
                "quick_quiz" => "/game/quick-quiz",
                "code_challenge" => "/game/code-challenge", 
                "scenario_simulator" => "/game/scenario-simulator",
                "speed_round" => "/game/speed-round",
                "skill_builder" => "/game/skill-builder",
                
                // 10 nuevos juegos universales
                "memory_challenge" => "/game/memory-challenge",
                "logic_puzzles" => "/game/logic-puzzles",
                "time_management" => "/game/time-management",
                "communication_skills" => "/game/communication-skills",
                "problem_solving_race" => "/game/problem-solving-race",
                "decision_trees" => "/game/decision-trees",
                "pattern_recognition" => "/game/pattern-recognition",
                "critical_thinking" => "/game/critical-thinking",
                "innovation_lab" => "/game/innovation-lab",
                "stress_test" => "/game/stress-test",
                
                _ => "/gamification" // Fallback a gamificación
            };
            
            // Navegar directamente (no nueva pestaña para mejor UX)
            Navigation.NavigateTo(route);
        }

        private async Task SwitchTab(string tab)
        {
            activeTab = tab;
            StateHasChanged();
            
            if (tab == "stats" && !chartsInitialized)
            {
                await Task.Delay(100); // Esperar a que se renderice el canvas
                await InitializeCharts();
            }
        }

        private async Task InitializeCharts()
        {
            try
            {
                // Inicializar gráficos de gamificación
                await JSRuntime.InvokeVoidAsync("initializeGamificationCharts");
                chartsInitialized = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing charts: {ex.Message}");
            }
        }

        private List<GameInfo> GetMockGames()
        {
            return new List<GameInfo>
            {
                // 5 juegos originales
                new GameInfo 
                { 
                    Id = "code_challenge_1", Name = "Desafío de Código", Type = "code_challenge", 
                    Description = "Resuelve problemas de programación", Stars = 4, AverageTimeMinutes = 15, PointsReward = 150, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "quick_quiz_1", Name = "Quiz Rápido", Type = "quick_quiz", 
                    Description = "Responde preguntas de conocimiento", Stars = 2, AverageTimeMinutes = 5, PointsReward = 80, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "scenario_simulator_1", Name = "Simulador de Escenarios", Type = "scenario_simulator", 
                    Description = "Enfrenta situaciones laborales reales", Stars = 5, AverageTimeMinutes = 20, PointsReward = 200, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "speed_round_1", Name = "Ronda Rápida", Type = "speed_round", 
                    Description = "Responde lo más rápido posible", Stars = 3, AverageTimeMinutes = 8, PointsReward = 120, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "skill_builder_1", Name = "Constructor de Habilidades", Type = "skill_builder", 
                    Description = "Mejora tus habilidades paso a paso", Stars = 4, AverageTimeMinutes = 12, PointsReward = 140, AiPowered = true
                },
                
                // 10 juegos universales nuevos
                new GameInfo 
                { 
                    Id = "memory_challenge_1", Name = "Desafío de Memoria", Type = "memory_challenge", 
                    Description = "Memoriza secuencias y patrones complejos", Stars = 3, AverageTimeMinutes = 6, PointsReward = 100, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "logic_puzzles_1", Name = "Rompecabezas Lógicos", Type = "logic_puzzles", 
                    Description = "Resuelve acertijos de lógica y razonamiento", Stars = 4, AverageTimeMinutes = 14, PointsReward = 160, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "time_management_1", Name = "Gestión del Tiempo", Type = "time_management", 
                    Description = "Prioriza tareas y gestiona deadlines", Stars = 3, AverageTimeMinutes = 10, PointsReward = 110, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "communication_skills_1", Name = "Habilidades de Comunicación", Type = "communication_skills", 
                    Description = "Practica comunicación efectiva", Stars = 3, AverageTimeMinutes = 12, PointsReward = 130, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "problem_solving_race_1", Name = "Carrera de Resolución", Type = "problem_solving_race", 
                    Description = "Resuelve problemas contra el tiempo", Stars = 4, AverageTimeMinutes = 9, PointsReward = 140, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "decision_trees_1", Name = "Árboles de Decisión", Type = "decision_trees", 
                    Description = "Toma decisiones estratégicas complejas", Stars = 5, AverageTimeMinutes = 18, PointsReward = 180, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "pattern_recognition_1", Name = "Reconocimiento de Patrones", Type = "pattern_recognition", 
                    Description = "Identifica patrones y tendencias", Stars = 3, AverageTimeMinutes = 7, PointsReward = 90, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "critical_thinking_1", Name = "Pensamiento Crítico", Type = "critical_thinking", 
                    Description = "Analiza información de forma crítica", Stars = 5, AverageTimeMinutes = 25, PointsReward = 220, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "innovation_lab_1", Name = "Laboratorio de Innovación", Type = "innovation_lab", 
                    Description = "Desarrolla soluciones creativas", Stars = 4, AverageTimeMinutes = 16, PointsReward = 170, AiPowered = true
                },
                new GameInfo 
                { 
                    Id = "stress_test_1", Name = "Prueba de Estrés", Type = "stress_test", 
                    Description = "Mantén el rendimiento bajo presión", Stars = 5, AverageTimeMinutes = 22, PointsReward = 250, AiPowered = true
                }
            };
        }

        private UserStats GetMockStats()
        {
            return new UserStats
            {
                UserId = userId,
                Level = 8,
                Experience = 7250,
                TotalPoints = 2850,
                TotalGamesPlayed = 47,
                TotalGamesWon = 32,
                StreakDays = 12,
                Rank = 156,
                GamesByType = new Dictionary<string, int>
                {
                    {"code_challenge", 8},
                    {"quick_quiz", 15},
                    {"scenario_simulator", 6},
                    {"speed_round", 10},
                    {"skill_builder", 8}
                },
                BestScores = new Dictionary<string, int>
                {
                    {"code_challenge", 95},
                    {"quick_quiz", 88},
                    {"scenario_simulator", 92}
                }
            };
        }

        private List<Achievement> GetMockAchievements()
        {
            return new List<Achievement>
            {
                // ============================================================================
                // LOGROS BÁSICOS - FÁCILES DE ALCANZAR
                // ============================================================================
                new Achievement
                {
                    Id = "first_game",
                    Name = "Primer Juego",
                    Description = "Completa tu primer desafío",
                    Icon = "🎮",
                    Points = 50,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-15),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "quick_starter",
                    Name = "Inicio Rápido",
                    Description = "Completa 3 juegos en tu primera sesión",
                    Icon = "🚀",
                    Points = 75,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-14),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "curious_mind",
                    Name = "Mente Curiosa",
                    Description = "Prueba 3 tipos diferentes de juegos",
                    Icon = "🧠",
                    Points = 100,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-12),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "persistent_player",
                    Name = "Jugador Persistente",
                    Description = "Juega durante 3 días consecutivos",
                    Icon = "🔥",
                    Points = 120,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-10),
                    Progress = 1.0f
                },

                // ============================================================================
                // LOGROS INTERMEDIOS - PROGRESO ACTUAL
                // ============================================================================
                new Achievement
                {
                    Id = "code_master",
                    Name = "Maestro del Código",
                    Description = "Completa 10 desafíos de código",
                    Icon = "💻",
                    Points = 200,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-8),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "memory_champion",
                    Name = "Campeón de Memoria",
                    Description = "Supera 15 desafíos de memoria",
                    Icon = "🧠",
                    Points = 180,
                    Unlocked = false,
                    Progress = 0.8f
                },
                new Achievement
                {
                    Id = "logic_master",
                    Name = "Maestro de la Lógica",
                    Description = "Resuelve 20 rompecabezas lógicos",
                    Icon = "🧩",
                    Points = 250,
                    Unlocked = false,
                    Progress = 0.7f
                },
                new Achievement
                {
                    Id = "time_manager",
                    Name = "Gestor del Tiempo",
                    Description = "Completa 12 ejercicios de gestión del tiempo",
                    Icon = "⏰",
                    Points = 200,
                    Unlocked = false,
                    Progress = 0.58f
                },
                new Achievement
                {
                    Id = "communicator",
                    Name = "Comunicador Efective",
                    Description = "Supera 10 ejercicios de comunicación",
                    Icon = "💬",
                    Points = 190,
                    Unlocked = false,
                    Progress = 0.5f
                },

                // ============================================================================
                // LOGROS AVANZADOS - EN PROGRESO TEMPRANO
                // ============================================================================
                new Achievement
                {
                    Id = "speed_demon",
                    Name = "Demonio de la Velocidad",
                    Description = "Completa 25 rondas rápidas",
                    Icon = "⚡",
                    Points = 300,
                    Unlocked = false,
                    Progress = 0.6f
                },
                new Achievement
                {
                    Id = "problem_solver",
                    Name = "Solucionador de Problemas",
                    Description = "Gana 30 carreras de resolución",
                    Icon = "🏃",
                    Points = 350,
                    Unlocked = false,
                    Progress = 0.4f
                },
                new Achievement
                {
                    Id = "decision_maker",
                    Name = "Tomador de Decisiones",
                    Description = "Completa 15 árboles de decisión",
                    Icon = "🌳",
                    Points = 280,
                    Unlocked = false,
                    Progress = 0.33f
                },
                new Achievement
                {
                    Id = "pattern_detector",
                    Name = "Detector de Patrones",
                    Description = "Identifica 50 patrones correctamente",
                    Icon = "👁️",
                    Points = 320,
                    Unlocked = false,
                    Progress = 0.42f
                },
                new Achievement
                {
                    Id = "critical_thinker",
                    Name = "Pensador Crítico",
                    Description = "Supera 20 desafíos de pensamiento crítico",
                    Icon = "🤔",
                    Points = 400,
                    Unlocked = false,
                    Progress = 0.35f
                },
                new Achievement
                {
                    Id = "innovator",
                    Name = "Innovador",
                    Description = "Completa 18 ejercicios de laboratorio de innovación",
                    Icon = "💡",
                    Points = 380,
                    Unlocked = false,
                    Progress = 0.28f
                },
                new Achievement
                {
                    Id = "stress_handler",
                    Name = "Manejador de Estrés",
                    Description = "Supera 12 pruebas de estrés",
                    Icon = "😰",
                    Points = 450,
                    Unlocked = false,
                    Progress = 0.25f
                },

                // ============================================================================
                // LOGROS DE CONSISTENCIA - ACTIVIDAD REGULAR
                // ============================================================================
                new Achievement
                {
                    Id = "weekly_warrior",
                    Name = "Guerrero Semanal",
                    Description = "Juega todos los días por una semana",
                    Icon = "🗓️",
                    Points = 250,
                    Unlocked = false,
                    Progress = 0.71f
                },
                new Achievement
                {
                    Id = "morning_person",
                    Name = "Persona Matutina",
                    Description = "Completa 10 sesiones antes de las 9 AM",
                    Icon = "🌅",
                    Points = 180,
                    Unlocked = false,
                    Progress = 0.3f
                },
                new Achievement
                {
                    Id = "night_owl",
                    Name = "Búho Nocturno",
                    Description = "Completa 8 sesiones después de las 10 PM",
                    Icon = "🌙",
                    Points = 160,
                    Unlocked = false,
                    Progress = 0.625f
                },

                // ============================================================================
                // LOGROS DE DOMINIO - OBJETIVOS A LARGO PLAZO
                // ============================================================================
                new Achievement
                {
                    Id = "game_master",
                    Name = "Maestro de Juegos",
                    Description = "Alcanza dominio en todos los tipos de juegos",
                    Icon = "👑",
                    Points = 1000,
                    Unlocked = false,
                    Progress = 0.2f
                },
                new Achievement
                {
                    Id = "perfectionist",
                    Name = "Perfeccionista",
                    Description = "Mantén 90%+ de precisión en 50 juegos",
                    Icon = "⭐",
                    Points = 800,
                    Unlocked = false,
                    Progress = 0.18f
                },
                new Achievement
                {
                    Id = "marathon_runner",
                    Name = "Corredor de Maratón",
                    Description = "Juega por más de 2 horas en una sesión",
                    Icon = "🏃‍♂️",
                    Points = 300,
                    Unlocked = false,
                    Progress = 0.65f
                },
                new Achievement
                {
                    Id = "centurion",
                    Name = "Centurión",
                    Description = "Completa 100 juegos en total",
                    Icon = "🛡️",
                    Points = 500,
                    Unlocked = false,
                    Progress = 0.47f
                },

                // ============================================================================
                // LOGROS ESPECIALES - EVENTOS Y HITOS
                // ============================================================================
                new Achievement
                {
                    Id = "first_week",
                    Name = "Primera Semana",
                    Description = "Sobrevive tu primera semana completa",
                    Icon = "🎉",
                    Points = 200,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-7),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "comeback_kid",
                    Name = "El Que Regresa",
                    Description = "Vuelve después de 3 días de inactividad",
                    Icon = "🔄",
                    Points = 150,
                    Unlocked = false,
                    Progress = 0.0f
                },
                new Achievement
                {
                    Id = "multitasker",
                    Name = "Multitarea",
                    Description = "Completa 3 tipos de juegos en una sola sesión",
                    Icon = "🎭",
                    Points = 220,
                    Unlocked = false,
                    Progress = 0.67f
                },
                new Achievement
                {
                    Id = "explorer",
                    Name = "Explorador",
                    Description = "Descubre todas las funciones de la plataforma",
                    Icon = "🗺️",
                    Points = 300,
                    Unlocked = false,
                    Progress = 0.85f
                },

                // ============================================================================
                // LOGROS SOCIALES - INTERACCIÓN Y COMUNIDAD  
                // ============================================================================
                new Achievement
                {
                    Id = "profile_complete",
                    Name = "Perfil Completo",
                    Description = "Completa todos los campos de tu perfil",
                    Icon = "📋",
                    Points = 100,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-5),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "skill_collector",
                    Name = "Coleccionista de Habilidades",
                    Description = "Añade al menos 10 habilidades a tu perfil",
                    Icon = "🎯",
                    Points = 150,
                    Unlocked = false,
                    Progress = 0.7f
                },

                // ============================================================================
                // LOGROS SECRETOS - DESCUBRIMIENTOS OCULTOS
                // ============================================================================
                new Achievement
                {
                    Id = "easter_egg_hunter",
                    Name = "Cazador de Huevos de Pascua",
                    Description = "Encuentra 3 secretos ocultos en la plataforma",
                    Icon = "🥚",
                    Points = 500,
                    Unlocked = false,
                    Progress = 0.33f
                },
                new Achievement
                {
                    Id = "bug_reporter",
                    Name = "Reportero de Bugs",
                    Description = "Reporta tu primer bug o sugerencia",
                    Icon = "🐛",
                    Points = 200,
                    Unlocked = false,
                    Progress = 0.0f
                }
            };
        }

        private List<LeaderboardEntry> GetMockLeaderboard()
        {
            return new List<LeaderboardEntry>
            {
                new LeaderboardEntry
                {
                    Rank = 1,
                    UserId = "user_1",
                    Username = "CodeMaster Pro",
                    TotalPoints = 5450,
                    Level = 15,
                    GamesWon = 89,
                    AchievementsCount = 24,
                    Profession = "Software Engineer"
                },
                new LeaderboardEntry
                {
                    Rank = 2,
                    UserId = "user_2",
                    Username = "Logic Queen",
                    TotalPoints = 4920,
                    Level = 13,
                    GamesWon = 76,
                    AchievementsCount = 22,
                    Profession = "Data Scientist"
                },
                new LeaderboardEntry
                {
                    Rank = 156,
                    UserId = userId,
                    Username = "Tú",
                    TotalPoints = 2850,
                    Level = 8,
                    GamesWon = 32,
                    AchievementsCount = 12,
                    Profession = "Developer"
                }
            };
        }

        /// <summary>
        /// Enriquece el leaderboard con los nombres reales de los usuarios desde la base de datos
        /// </summary>
        private async Task EnrichLeaderboardWithRealNames(List<LeaderboardEntry> leaderboard)
        {
            try
            {
                using var db = await DbFactory.CreateDbContextAsync();
                
                foreach (var entry in leaderboard)
                {
                    // El user_id viene en formato "user_email_at_domain" o "user-xxxxx"
                    // Necesitamos extraer el email o buscar por el formato
                    string? email = null;
                    
                    if (entry.UserId.StartsWith("user_"))
                    {
                        // Formato: user_email_at_domain
                        // Convertir de vuelta: user_email_at_domain -> email@domain.com
                        var emailPart = entry.UserId.Replace("user_", "");
                        email = emailPart.Replace("_at_", "@").Replace("_", ".");
                    }
                    else if (entry.UserId.StartsWith("user-"))
                    {
                        // Formato: user-xxxxx (GUID), buscar por ID numérico si es posible
                        // Intentar buscar por el ID numérico si existe
                        var idPart = entry.UserId.Replace("user-", "");
                        if (int.TryParse(idPart, out var userId))
                        {
                            var user = await db.Users.FindAsync(userId);
                            if (user != null)
                            {
                                entry.Username = $"{user.Name} {user.LastName}".Trim();
                                continue;
                            }
                        }
                        // Si no se puede parsear, continuar con el siguiente
                        continue;
                    }
                    
                    // Buscar usuario por email
                    if (!string.IsNullOrEmpty(email))
                    {
                        var user = await db.Users.FirstOrDefaultAsync(u => u.Email == email);
                        if (user != null)
                        {
                            // Usar nombre completo o solo nombre si no hay apellido
                            var fullName = $"{user.Name} {user.LastName}".Trim();
                            if (string.IsNullOrWhiteSpace(fullName))
                            {
                                fullName = user.Name;
                            }
                            if (string.IsNullOrWhiteSpace(fullName))
                            {
                                fullName = user.Email.Split('@')[0]; // Usar parte antes del @ como fallback
                            }
                            entry.Username = fullName;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error enriching leaderboard with real names: {ex.Message}");
                // Continuar sin enriquecer si hay error
            }
        }
    }
}

