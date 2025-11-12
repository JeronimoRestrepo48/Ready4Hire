using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Data
{
    public static class BadgeSeeder
    {
        public static List<Badge> GetInitialBadges()
        {
            return new List<Badge>
            {
                // ============================================================================
                // BADGES DE INICIO - COMMON
                // ============================================================================
                new Badge
                {
                    Name = "🎬 Primer Paso",
                    Description = "Completa tu primera entrevista",
                    Icon = "🎬",
                    Category = "milestone",
                    Rarity = "common",
                    RequirementType = "interviews_completed",
                    RequirementValue = 1,
                    RewardPoints = 50,
                    RewardXp = 100
                },
                new Badge
                {
                    Name = "🎮 Jugador Novato",
                    Description = "Juega tu primer juego de práctica",
                    Icon = "🎮",
                    Category = "general",
                    Rarity = "common",
                    RequirementType = "games_played",
                    RequirementValue = 1,
                    RewardPoints = 30,
                    RewardXp = 50
                },
                new Badge
                {
                    Name = "📚 Estudiante Dedicado",
                    Description = "Completa 5 entrevistas de práctica",
                    Icon = "📚",
                    Category = "milestone",
                    Rarity = "common",
                    RequirementType = "interviews_completed",
                    RequirementValue = 5,
                    RewardPoints = 100,
                    RewardXp = 200
                },

                // ============================================================================
                // BADGES TÉCNICOS - RARE
                // ============================================================================
                new Badge
                {
                    Name = "💻 Código Maestro",
                    Description = "Gana 10 desafíos de código",
                    Icon = "💻",
                    Category = "technical",
                    Rarity = "rare",
                    RequirementType = "code_challenges_won",
                    RequirementValue = 10,
                    RewardPoints = 300,
                    RewardXp = 500
                },
                new Badge
                {
                    Name = "🧠 Cerebro Técnico",
                    Description = "Responde 50 preguntas técnicas correctamente",
                    Icon = "🧠",
                    Category = "technical",
                    Rarity = "rare",
                    RequirementType = "technical_correct_answers",
                    RequirementValue = 50,
                    RewardPoints = 400,
                    RewardXp = 600
                },
                new Badge
                {
                    Name = "⚡ Velocista",
                    Description = "Completa 5 Speed Rounds con 90%+ precisión",
                    Icon = "⚡",
                    Category = "technical",
                    Rarity = "rare",
                    RequirementType = "speed_rounds_90_accuracy",
                    RequirementValue = 5,
                    RewardPoints = 350,
                    RewardXp = 550
                },

                // ============================================================================
                // BADGES SOFT SKILLS - RARE
                // ============================================================================
                new Badge
                {
                    Name = "💬 Comunicador Experto",
                    Description = "Completa 10 entrevistas de soft skills con excelencia",
                    Icon = "💬",
                    Category = "soft_skills",
                    Rarity = "rare",
                    RequirementType = "soft_skills_interviews",
                    RequirementValue = 10,
                    RewardPoints = 300,
                    RewardXp = 500
                },
                new Badge
                {
                    Name = "🎯 Solucionador de Problemas",
                    Description = "Resuelve 15 simulaciones de escenarios",
                    Icon = "🎯",
                    Category = "soft_skills",
                    Rarity = "rare",
                    RequirementType = "scenarios_completed",
                    RequirementValue = 15,
                    RewardPoints = 350,
                    RewardXp = 550
                },

                // ============================================================================
                // BADGES DE LOGROS - EPIC
                // ============================================================================
                new Badge
                {
                    Name = "🔥 Racha Imparable",
                    Description = "Mantén una racha de 7 días consecutivos",
                    Icon = "🔥",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "streak_days",
                    RequirementValue = 7,
                    RewardPoints = 500,
                    RewardXp = 800
                },
                new Badge
                {
                    Name = "🏆 Campeón",
                    Description = "Gana 50 juegos en total",
                    Icon = "🏆",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "games_won",
                    RequirementValue = 50,
                    RewardPoints = 600,
                    RewardXp = 1000
                },
                new Badge
                {
                    Name = "🎓 Experto",
                    Description = "Alcanza el nivel 10",
                    Icon = "🎓",
                    Category = "milestone",
                    Rarity = "epic",
                    RequirementType = "level_reached",
                    RequirementValue = 10,
                    RewardPoints = 1000,
                    RewardXp = 1500
                },
                new Badge
                {
                    Name = "⭐ Perfeccionista",
                    Description = "Consigue 100% de precisión en 5 entrevistas",
                    Icon = "⭐",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "perfect_interviews",
                    RequirementValue = 5,
                    RewardPoints = 800,
                    RewardXp = 1200
                },

                // ============================================================================
                // BADGES LEGENDARIOS - LEGENDARY
                // ============================================================================
                new Badge
                {
                    Name = "👑 Maestro Supremo",
                    Description = "Alcanza el nivel 25",
                    Icon = "👑",
                    Category = "milestone",
                    Rarity = "legendary",
                    RequirementType = "level_reached",
                    RequirementValue = 25,
                    RewardPoints = 2500,
                    RewardXp = 5000
                },
                new Badge
                {
                    Name = "💎 Diamante",
                    Description = "Acumula 10,000 puntos totales",
                    Icon = "💎",
                    Category = "achievement",
                    Rarity = "legendary",
                    RequirementType = "total_points",
                    RequirementValue = 10000,
                    RewardPoints = 3000,
                    RewardXp = 6000
                },
                new Badge
                {
                    Name = "🌟 Leyenda",
                    Description = "Completa 100 entrevistas exitosamente",
                    Icon = "🌟",
                    Category = "milestone",
                    Rarity = "legendary",
                    RequirementType = "interviews_completed",
                    RequirementValue = 100,
                    RewardPoints = 5000,
                    RewardXp = 10000
                },
                new Badge
                {
                    Name = "🔥 Racha Épica",
                    Description = "Mantén una racha de 30 días consecutivos",
                    Icon = "🔥",
                    Category = "achievement",
                    Rarity = "legendary",
                    RequirementType = "streak_days",
                    RequirementValue = 30,
                    RewardPoints = 4000,
                    RewardXp = 8000
                },

                // ============================================================================
                // BADGES ESPECIALES - EPIC
                // ============================================================================
                new Badge
                {
                    Name = "🌙 Búho Nocturno",
                    Description = "Completa 10 sesiones entre 10 PM y 6 AM",
                    Icon = "🌙",
                    Category = "general",
                    Rarity = "epic",
                    RequirementType = "night_sessions",
                    RequirementValue = 10,
                    RewardPoints = 400,
                    RewardXp = 700
                },
                new Badge
                {
                    Name = "☀️ Madrugador",
                    Description = "Completa 10 sesiones antes de las 8 AM",
                    Icon = "☀️",
                    Category = "general",
                    Rarity = "epic",
                    RequirementType = "morning_sessions",
                    RequirementValue = 10,
                    RewardPoints = 400,
                    RewardXp = 700
                },
                new Badge
                {
                    Name = "🚀 Velocidad Supersónica",
                    Description = "Completa una entrevista en menos de 15 minutos con 95%+",
                    Icon = "🚀",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "fast_interview_95_accuracy",
                    RequirementValue = 1,
                    RewardPoints = 700,
                    RewardXp = 1100
                },
                new Badge
                {
                    Name = "🎨 Innovador",
                    Description = "Prueba todos los tipos de juegos disponibles",
                    Icon = "🎨",
                    Category = "general",
                    Rarity = "rare",
                    RequirementType = "all_game_types_played",
                    RequirementValue = 6,
                    RewardPoints = 300,
                    RewardXp = 500
                },
                new Badge
                {
                    Name = "🌍 Políglota",
                    Description = "Completa entrevistas en 3 idiomas diferentes",
                    Icon = "🌍",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "languages_used",
                    RequirementValue = 3,
                    RewardPoints = 600,
                    RewardXp = 900
                },

                // ============================================================================
                // NUEVAS BADGES COMÚN - MÁS ALCANZABLES
                // ============================================================================
                new Badge
                {
                    Name = "🏃 Principiante Activo",
                    Description = "Juega 3 juegos en un día",
                    Icon = "🏃",
                    Category = "general",
                    Rarity = "common",
                    RequirementType = "daily_games_played",
                    RequirementValue = 3,
                    RewardPoints = 50,
                    RewardXp = 80
                },
                new Badge
                {
                    Name = "🔄 Perseverante",
                    Description = "Reintenta un juego después de fallar",
                    Icon = "🔄",
                    Category = "general",
                    Rarity = "common",
                    RequirementType = "games_retried",
                    RequirementValue = 1,
                    RewardPoints = 40,
                    RewardXp = 60
                },
                new Badge
                {
                    Name = "📝 Completista",
                    Description = "Completa tu perfil al 100%",
                    Icon = "📝",
                    Category = "milestone",
                    Rarity = "common",
                    RequirementType = "profile_completion",
                    RequirementValue = 100,
                    RewardPoints = 100,
                    RewardXp = 150
                },
                new Badge
                {
                    Name = "🎲 Explorador",
                    Description = "Prueba 3 tipos diferentes de juegos",
                    Icon = "🎲",
                    Category = "general",
                    Rarity = "common",
                    RequirementType = "different_game_types",
                    RequirementValue = 3,
                    RewardPoints = 75,
                    RewardXp = 120
                },
                new Badge
                {
                    Name = "⏰ Puntual",
                    Description = "Completa una sesión en menos de 30 minutos",
                    Icon = "⏰",
                    Category = "general",
                    Rarity = "common",
                    RequirementType = "quick_session",
                    RequirementValue = 1,
                    RewardPoints = 60,
                    RewardXp = 90
                },

                // ============================================================================
                // NUEVAS BADGES RARAS - INTERMEDIAS
                // ============================================================================
                new Badge
                {
                    Name = "🧩 Solucionador Nato",
                    Description = "Resuelve 25 rompecabezas lógicos",
                    Icon = "🧩",
                    Category = "technical",
                    Rarity = "rare",
                    RequirementType = "logic_puzzles_solved",
                    RequirementValue = 25,
                    RewardPoints = 250,
                    RewardXp = 400
                },
                new Badge
                {
                    Name = "💭 Pensador Crítico",
                    Description = "Completa 15 desafíos de pensamiento crítico",
                    Icon = "💭",
                    Category = "soft_skills",
                    Rarity = "rare",
                    RequirementType = "critical_thinking_completed",
                    RequirementValue = 15,
                    RewardPoints = 300,
                    RewardXp = 450
                },
                new Badge
                {
                    Name = "🏋️ Entrenador Mental",
                    Description = "Completa 20 desafíos de memoria",
                    Icon = "🏋️",
                    Category = "technical",
                    Rarity = "rare",
                    RequirementType = "memory_challenges_completed",
                    RequirementValue = 20,
                    RewardPoints = 280,
                    RewardXp = 420
                },
                new Badge
                {
                    Name = "⚖️ Organizador Experto",
                    Description = "Completa 12 ejercicios de gestión del tiempo",
                    Icon = "⚖️",
                    Category = "soft_skills",
                    Rarity = "rare",
                    RequirementType = "time_management_completed",
                    RequirementValue = 12,
                    RewardPoints = 320,
                    RewardXp = 480
                },
                new Badge
                {
                    Name = "🗣️ Comunicador Eficaz",
                    Description = "Completa 18 ejercicios de comunicación",
                    Icon = "🗣️",
                    Category = "soft_skills",
                    Rarity = "rare",
                    RequirementType = "communication_exercises_completed",
                    RequirementValue = 18,
                    RewardPoints = 290,
                    RewardXp = 440
                },
                new Badge
                {
                    Name = "🎯 Precisión Láser",
                    Description = "Mantén 85%+ de precisión en 10 juegos seguidos",
                    Icon = "🎯",
                    Category = "achievement",
                    Rarity = "rare",
                    RequirementType = "accuracy_streak",
                    RequirementValue = 10,
                    RewardPoints = 400,
                    RewardXp = 600
                },
                new Badge
                {
                    Name = "🔥 Semana Intensiva",
                    Description = "Juega todos los días de una semana",
                    Icon = "🔥",
                    Category = "achievement",
                    Rarity = "rare",
                    RequirementType = "weekly_streak",
                    RequirementValue = 7,
                    RewardPoints = 350,
                    RewardXp = 500
                },

                // ============================================================================
                // NUEVAS BADGES ÉPICAS - DESAFIANTES PERO ALCANZABLES
                // ============================================================================
                new Badge
                {
                    Name = "🎨 Creativo Innovador",
                    Description = "Completa 25 ejercicios del laboratorio de innovación",
                    Icon = "🎨",
                    Category = "soft_skills",
                    Rarity = "epic",
                    RequirementType = "innovation_lab_completed",
                    RequirementValue = 25,
                    RewardPoints = 500,
                    RewardXp = 750
                },
                new Badge
                {
                    Name = "🏃‍♂️ Corredor de Problemas",
                    Description = "Gana 30 carreras de resolución de problemas",
                    Icon = "🏃‍♂️", 
                    Category = "technical",
                    Rarity = "epic",
                    RequirementType = "problem_solving_races_won",
                    RequirementValue = 30,
                    RewardPoints = 600,
                    RewardXp = 900
                },
                new Badge
                {
                    Name = "🌳 Estratega Maestro",
                    Description = "Completa 20 árboles de decisión complejos",
                    Icon = "🌳",
                    Category = "soft_skills",
                    Rarity = "epic",
                    RequirementType = "decision_trees_completed",
                    RequirementValue = 20,
                    RewardPoints = 550,
                    RewardXp = 800
                },
                new Badge
                {
                    Name = "👁️ Detector de Patrones",
                    Description = "Identifica correctamente 100 patrones",
                    Icon = "👁️",
                    Category = "technical",
                    Rarity = "epic",
                    RequirementType = "patterns_identified",
                    RequirementValue = 100,
                    RewardPoints = 700,
                    RewardXp = 1000
                },
                new Badge
                {
                    Name = "💪 Resistente al Estrés",
                    Description = "Supera 15 pruebas de estrés exitosamente",
                    Icon = "💪",
                    Category = "soft_skills",
                    Rarity = "epic",
                    RequirementType = "stress_tests_passed",
                    RequirementValue = 15,
                    RewardPoints = 650,
                    RewardXp = 950
                },
                new Badge
                {
                    Name = "🎖️ Veterano",
                    Description = "Lleva más de 60 días registrado",
                    Icon = "🎖️",
                    Category = "milestone",
                    Rarity = "epic",
                    RequirementType = "days_registered",
                    RequirementValue = 60,
                    RewardPoints = 500,
                    RewardXp = 750
                },
                new Badge
                {
                    Name = "🔥 Racha de Oro",
                    Description = "Mantén una racha de 14 días consecutivos",
                    Icon = "🔥",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "streak_days",
                    RequirementValue = 14,
                    RewardPoints = 800,
                    RewardXp = 1200
                },
                new Badge
                {
                    Name = "🏆 Ganador Consistente",
                    Description = "Gana 100 juegos en total",
                    Icon = "🏆",
                    Category = "achievement",
                    Rarity = "epic",
                    RequirementType = "games_won",
                    RequirementValue = 100,
                    RewardPoints = 750,
                    RewardXp = 1100
                },

                // ============================================================================
                // NUEVAS BADGES LEGENDARIAS - OBJETIVOS A LARGO PLAZO
                // ============================================================================
                new Badge
                {
                    Name = "🧠 Genio Multidisciplinario",
                    Description = "Alcanza maestría en todos los tipos de juegos",
                    Icon = "🧠",
                    Category = "achievement",
                    Rarity = "legendary",
                    RequirementType = "mastery_all_games",
                    RequirementValue = 15,
                    RewardPoints = 2000,
                    RewardXp = 3000
                },
                new Badge
                {
                    Name = "🎭 Camaleón Profesional",
                    Description = "Completa entrevistas para 5 profesiones diferentes",
                    Icon = "🎭",
                    Category = "milestone",
                    Rarity = "legendary",
                    RequirementType = "different_professions",
                    RequirementValue = 5,
                    RewardPoints = 1800,
                    RewardXp = 2700
                },
                new Badge
                {
                    Name = "⚡ Rayo Humano",
                    Description = "Completa 50 rondas rápidas en tiempo récord",
                    Icon = "⚡",
                    Category = "achievement",
                    Rarity = "legendary",
                    RequirementType = "speed_rounds_record",
                    RequirementValue = 50,
                    RewardPoints = 2200,
                    RewardXp = 3300
                },
                new Badge
                {
                    Name = "🌟 Mentor de la Comunidad",
                    Description = "Ayuda a otros usuarios conseguir sus primeros logros",
                    Icon = "🌟",
                    Category = "general",
                    Rarity = "legendary",
                    RequirementType = "mentoring_achievements",
                    RequirementValue = 10,
                    RewardPoints = 3000,
                    RewardXp = 5000
                },
                new Badge
                {
                    Name = "🏰 Constructor de Imperio",
                    Description = "Acumula más de 50,000 puntos de experiencia",
                    Icon = "🏰",
                    Category = "milestone",
                    Rarity = "legendary",
                    RequirementType = "total_experience",
                    RequirementValue = 50000,
                    RewardPoints = 4000,
                    RewardXp = 7500
                },
                new Badge
                {
                    Name = "🎯 Perfección Absoluta",
                    Description = "Mantén 100% de precisión en 20 sesiones completas",
                    Icon = "🎯",
                    Category = "achievement",
                    Rarity = "legendary",
                    RequirementType = "perfect_sessions",
                    RequirementValue = 20,
                    RewardPoints = 5000,
                    RewardXp = 8000
                },

                // ============================================================================
                // BADGES ESPECIALES Y SECRETAS
                // ============================================================================
                new Badge
                {
                    Name = "🎂 Primer Aniversario",
                    Description = "Celebra un año completo en Ready4Hire",
                    Icon = "🎂",
                    Category = "milestone",
                    Rarity = "legendary",
                    RequirementType = "days_registered",
                    RequirementValue = 365,
                    RewardPoints = 3650,
                    RewardXp = 5000
                },
                new Badge
                {
                    Name = "🎃 Cazador Nocturno",
                    Description = "Completa 25 sesiones entre medianoche y 6 AM",
                    Icon = "🎃",
                    Category = "general",
                    Rarity = "epic",
                    RequirementType = "midnight_sessions",
                    RequirementValue = 25,
                    RewardPoints = 600,
                    RewardXp = 900
                },
                new Badge
                {
                    Name = "⚖️ Equilibrio Perfecto",
                    Description = "Mantén el mismo número de juegos técnicos y soft skills",
                    Icon = "⚖️",
                    Category = "achievement",
                    Rarity = "rare",
                    RequirementType = "balanced_gameplay",
                    RequirementValue = 50,
                    RewardPoints = 400,
                    RewardXp = 600
                },
                new Badge
                {
                    Name = "🔍 Inspector",
                    Description = "Encuentra y reporta 3 bugs o mejoras",
                    Icon = "🔍",
                    Category = "general",
                    Rarity = "epic",
                    RequirementType = "bugs_reported",
                    RequirementValue = 3,
                    RewardPoints = 800,
                    RewardXp = 1200
                },
                new Badge
                {
                    Name = "💝 Embajador",
                    Description = "Invita a 5 amigos a unirse a Ready4Hire",
                    Icon = "💝",
                    Category = "general",
                    Rarity = "rare",
                    RequirementType = "referrals_successful",
                    RequirementValue = 5,
                    RewardPoints = 500,
                    RewardXp = 750
                }
            };
        }
    }
}

