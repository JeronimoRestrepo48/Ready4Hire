using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Services
{
    public class AchievementProgressService
    {
        /// <summary>
        /// Actualiza el progreso de un logro específico para un usuario
        /// </summary>
        public async Task<bool> UpdateAchievementProgressAsync(string userId, string achievementId, float progressIncrement = 0.1f)
        {
            try
            {
                // Por ahora simular actualización de progreso
                // En el futuro esto se conectará con una base de datos real
                await Task.Delay(100); // Simular operación async
                
                Console.WriteLine($"Achievement progress updated: User {userId}, Achievement {achievementId}, Progress +{progressIncrement}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating achievement progress: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Verifica si un logro debería desbloquearse
        /// </summary>
        public async Task<bool> CheckAchievementUnlockAsync(string userId, string achievementId)
        {
            try
            {
                // Simular verificación de desbloqueo
                await Task.Delay(50);
                
                // Por ahora usar probabilidad aleatoria para simular desbloqueos
                var shouldUnlock = Random.Shared.NextDouble() > 0.8; // 20% chance
                
                if (shouldUnlock)
                {
                    Console.WriteLine($"Achievement unlocked! User {userId}, Achievement {achievementId}");
                }
                
                return shouldUnlock;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error checking achievement unlock: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Actualiza múltiples logros basados en una acción del usuario
        /// </summary>
        public async Task UpdateAchievementsForActionAsync(string userId, string actionType, Dictionary<string, object>? parameters = null)
        {
            try
            {
                var achievementsToUpdate = GetAchievementsForAction(actionType);
                
                foreach (var achievementId in achievementsToUpdate)
                {
                    await UpdateAchievementProgressAsync(userId, achievementId);
                    await CheckAchievementUnlockAsync(userId, achievementId);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error updating achievements for action {actionType}: {ex.Message}");
            }
        }

        /// <summary>
        /// Obtiene los logros que deberían actualizarse para una acción específica
        /// </summary>
        private List<string> GetAchievementsForAction(string actionType)
        {
            return actionType.ToLower() switch
            {
                "game_completed" => new List<string> 
                { 
                    "first_game", "quick_starter", "persistent_player", "centurion", "weekly_warrior" 
                },
                "code_challenge_completed" => new List<string> 
                { 
                    "code_master", "problem_solver", "perfectionist" 
                },
                "memory_game_completed" => new List<string> 
                { 
                    "memory_champion", "game_master" 
                },
                "logic_puzzle_completed" => new List<string> 
                { 
                    "logic_master", "critical_thinker", "game_master" 
                },
                "speed_round_completed" => new List<string> 
                { 
                    "speed_demon", "perfectionist", "marathon_runner" 
                },
                "communication_exercise_completed" => new List<string> 
                { 
                    "communicator", "game_master" 
                },
                "time_management_completed" => new List<string> 
                { 
                    "time_manager", "game_master" 
                },
                "daily_login" => new List<string> 
                { 
                    "persistent_player", "weekly_warrior" 
                },
                "morning_session" => new List<string> 
                { 
                    "morning_person" 
                },
                "night_session" => new List<string> 
                { 
                    "night_owl" 
                },
                "profile_updated" => new List<string> 
                { 
                    "profile_complete", "skill_collector", "explorer" 
                },
                "perfect_score" => new List<string> 
                { 
                    "perfectionist", "game_master" 
                },
                "long_session" => new List<string> 
                { 
                    "marathon_runner" 
                },
                "game_types_explored" => new List<string> 
                { 
                    "curious_mind", "multitasker", "explorer", "game_master" 
                },
                "bug_reported" => new List<string> 
                { 
                    "bug_reporter", "explorer" 
                },
                "easter_egg_found" => new List<string> 
                { 
                    "easter_egg_hunter", "explorer" 
                },
                _ => new List<string>()
            };
        }

        /// <summary>
        /// Simula el progreso actual de logros para un usuario
        /// </summary>
        public async Task<List<Achievement>> GetUserAchievementsAsync(string userId)
        {
            // Por ahora retornar los logros mock
            // En el futuro esto consultará la base de datos real
            await Task.Delay(100);
            
            return GetSimulatedAchievements();
        }

        /// <summary>
        /// Simula logros con progreso realista
        /// </summary>
        private List<Achievement> GetSimulatedAchievements()
        {
            var random = new Random();
            
            return new List<Achievement>
            {
                new Achievement
                {
                    Id = "first_game",
                    Name = "Primer Juego",
                    Description = "Completa tu primer desafío",
                    Icon = "🎮",
                    Points = 50,
                    Unlocked = true,
                    UnlockedAt = DateTime.Now.AddDays(-random.Next(1, 15)),
                    Progress = 1.0f
                },
                new Achievement
                {
                    Id = "code_master",
                    Name = "Maestro del Código",
                    Description = "Completa 10 desafíos de código",
                    Icon = "💻",
                    Points = 200,
                    Unlocked = random.NextDouble() > 0.6,
                    UnlockedAt = random.NextDouble() > 0.6 ? DateTime.Now.AddDays(-random.Next(1, 10)) : null,
                    Progress = (float)(random.NextDouble() * 0.4 + 0.6) // 60-100%
                },
                new Achievement
                {
                    Id = "memory_champion",
                    Name = "Campeón de Memoria",
                    Description = "Supera 15 desafíos de memoria",
                    Icon = "🧠",
                    Points = 180,
                    Unlocked = false,
                    Progress = (float)(random.NextDouble() * 0.5 + 0.3) // 30-80%
                },
                new Achievement
                {
                    Id = "speed_demon",
                    Name = "Demonio de la Velocidad",
                    Description = "Completa 25 rondas rápidas",
                    Icon = "⚡",
                    Points = 300,
                    Unlocked = false,
                    Progress = (float)(random.NextDouble() * 0.7) // 0-70%
                },
                new Achievement
                {
                    Id = "perfectionist",
                    Name = "Perfeccionista",
                    Description = "Mantén 90%+ de precisión en 50 juegos",
                    Icon = "⭐",
                    Points = 800,
                    Unlocked = false,
                    Progress = (float)(random.NextDouble() * 0.3) // 0-30%
                }
            };
        }

        /// <summary>
        /// Trigger para actualizar logros cuando se completa un juego
        /// </summary>
        public async Task OnGameCompletedAsync(string userId, string gameType, bool won, float accuracy, TimeSpan duration)
        {
            var tasks = new List<Task>();

            // Actualizar logros base
            tasks.Add(UpdateAchievementsForActionAsync(userId, "game_completed"));

            // Actualizar logros específicos del tipo de juego
            tasks.Add(UpdateAchievementsForActionAsync(userId, $"{gameType}_completed"));

            // Actualizar logros de tiempo
            var currentHour = DateTime.Now.Hour;
            if (currentHour < 9)
                tasks.Add(UpdateAchievementsForActionAsync(userId, "morning_session"));
            else if (currentHour >= 22)
                tasks.Add(UpdateAchievementsForActionAsync(userId, "night_session"));

            // Actualizar logros de precisión
            if (accuracy >= 0.9f)
                tasks.Add(UpdateAchievementsForActionAsync(userId, "perfect_score"));

            // Actualizar logros de duración
            if (duration.TotalMinutes >= 120)
                tasks.Add(UpdateAchievementsForActionAsync(userId, "long_session"));

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Trigger para actualizar logros cuando el usuario inicia sesión
        /// </summary>
        public async Task OnUserLoginAsync(string userId)
        {
            await UpdateAchievementsForActionAsync(userId, "daily_login");
        }

        /// <summary>
        /// Trigger para actualizar logros cuando se actualiza el perfil
        /// </summary>
        public async Task OnProfileUpdatedAsync(string userId)
        {
            await UpdateAchievementsForActionAsync(userId, "profile_updated");
        }
    }
}
