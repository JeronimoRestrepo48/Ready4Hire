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
            };
        }
    }
}

