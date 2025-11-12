using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Services
{
    public class BadgeProgressService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;

        public BadgeProgressService(IDbContextFactory<AppDbContext> dbFactory)
        {
            _dbFactory = dbFactory;
        }

        /// <summary>
        /// Actualiza el progreso de todas las insignias de un usuario
        /// </summary>
        public async Task UpdateAllBadgeProgressAsync(int userId)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            
            var user = await db.Users
                .Include(u => u.Badges)
                .ThenInclude(ub => ub.Badge)
                .FirstOrDefaultAsync(u => u.Id == userId);

            if (user == null) return;

            var allBadges = await db.Badges.Where(b => b.IsActive).ToListAsync();
            
            foreach (var badge in allBadges)
            {
                var userBadge = user.Badges.FirstOrDefault(ub => ub.BadgeId == badge.Id);
                if (userBadge == null)
                {
                    // Crear nuevo UserBadge si no existe
                    userBadge = new UserBadge 
                    { 
                        UserId = userId, 
                        BadgeId = badge.Id, 
                        IsUnlocked = false, 
                        Progress = 0 
                    };
                    db.UserBadges.Add(userBadge);
                    user.Badges.Add(userBadge);
                }

                if (!userBadge.IsUnlocked)
                {
                    var (progress, isUnlocked) = await CalculateBadgeProgressAsync(user, badge);
                    userBadge.Progress = progress;
                    
                    if (isUnlocked && !userBadge.IsUnlocked)
                    {
                        userBadge.IsUnlocked = true;
                        userBadge.UnlockedAt = DateTime.UtcNow;
                        
                        // Otorgar recompensas
                        user.TotalPoints += badge.RewardPoints;
                        user.Experience += badge.RewardXp;
                        
                        // Actualizar nivel basado en experiencia
                        UpdateUserLevel(user);
                    }
                }
            }

            await db.SaveChangesAsync();
        }

        /// <summary>
        /// Calcula el progreso específico de una insignia para un usuario
        /// </summary>
        private async Task<(float progress, bool isUnlocked)> CalculateBadgeProgressAsync(User user, Badge badge)
        {
            if (string.IsNullOrEmpty(badge.RequirementType) || badge.RequirementValue <= 0)
                return (0f, false);

            using var db = await _dbFactory.CreateDbContextAsync();
            float currentValue = 0;

            try
            {
                currentValue = badge.RequirementType switch
                {
                    // Logros básicos
                    "interviews_completed" => await GetInterviewsCompletedAsync(db, user.Id),
                    "games_played" => user.TotalGamesPlayed,
                    "games_won" => user.TotalGamesWon,
                    "streak_days" => user.StreakDays,
                    "level_reached" => user.Level,
                    "total_points" => user.TotalPoints,
                    "total_experience" => user.Experience,
                    
                    // Logros técnicos
                    "code_challenges_won" => await GetGameTypeWinsAsync(db, user.Id, "code_challenge"),
                    "technical_correct_answers" => await GetTechnicalAnswersAsync(db, user.Id),
                    "speed_rounds_90_accuracy" => await GetSpeedRoundsHighAccuracyAsync(db, user.Id),
                    "memory_challenges_completed" => await GetGameTypeCompletedAsync(db, user.Id, "memory_challenge"),
                    "logic_puzzles_solved" => await GetGameTypeCompletedAsync(db, user.Id, "logic_puzzles"),
                    "problem_solving_races_won" => await GetGameTypeWinsAsync(db, user.Id, "problem_solving_race"),
                    "patterns_identified" => await GetPatternsIdentifiedAsync(db, user.Id),
                    
                    // Logros soft skills
                    "soft_skills_interviews" => await GetSoftSkillsInterviewsAsync(db, user.Id),
                    "scenarios_completed" => await GetGameTypeCompletedAsync(db, user.Id, "scenario_simulator"),
                    "communication_exercises_completed" => await GetGameTypeCompletedAsync(db, user.Id, "communication_skills"),
                    "time_management_completed" => await GetGameTypeCompletedAsync(db, user.Id, "time_management"),
                    "critical_thinking_completed" => await GetGameTypeCompletedAsync(db, user.Id, "critical_thinking"),
                    "decision_trees_completed" => await GetGameTypeCompletedAsync(db, user.Id, "decision_trees"),
                    "innovation_lab_completed" => await GetGameTypeCompletedAsync(db, user.Id, "innovation_lab"),
                    "stress_tests_passed" => await GetGameTypeWinsAsync(db, user.Id, "stress_test"),
                    
                    // Logros de actividad
                    "daily_games_played" => await GetDailyGamesPlayedAsync(db, user.Id),
                    "games_retried" => await GetGamesRetriedAsync(db, user.Id),
                    "different_game_types" => await GetDifferentGameTypesPlayedAsync(db, user.Id),
                    "quick_session" => await GetQuickSessionsAsync(db, user.Id),
                    "accuracy_streak" => await GetAccuracyStreakAsync(db, user.Id),
                    "weekly_streak" => await GetWeeklyStreakAsync(db, user.Id),
                    "night_sessions" => await GetNightSessionsAsync(db, user.Id),
                    "morning_sessions" => await GetMorningSessionsAsync(db, user.Id),
                    "midnight_sessions" => await GetMidnightSessionsAsync(db, user.Id),
                    
                    // Logros especiales
                    "all_game_types_played" => await GetAllGameTypesPlayedAsync(db, user.Id),
                    "perfect_interviews" => await GetPerfectInterviewsAsync(db, user.Id),
                    "perfect_sessions" => await GetPerfectSessionsAsync(db, user.Id),
                    "fast_interview_95_accuracy" => await GetFastHighAccuracyInterviewsAsync(db, user.Id),
                    "mastery_all_games" => await GetGameMasteryAsync(db, user.Id),
                    "different_professions" => await GetDifferentProfessionsAsync(db, user.Id),
                    "speed_rounds_record" => await GetSpeedRoundsRecordAsync(db, user.Id),
                    "balanced_gameplay" => await GetBalancedGameplayAsync(db, user.Id),
                    
                    // Logros temporales y sociales
                    "days_registered" => (DateTime.UtcNow - (user.LastActivityDate ?? DateTime.UtcNow.AddDays(-30))).Days,
                    "profile_completion" => GetProfileCompletionPercentage(user),
                    "languages_used" => await GetLanguagesUsedAsync(db, user.Id),
                    "mentoring_achievements" => await GetMentoringAchievementsAsync(db, user.Id),
                    "bugs_reported" => await GetBugsReportedAsync(db, user.Id),
                    "referrals_successful" => await GetSuccessfulReferralsAsync(db, user.Id),
                    
                    _ => 0f
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error calculating badge progress for {badge.RequirementType}: {ex.Message}");
                return (0f, false);
            }

            var progress = Math.Min(currentValue / badge.RequirementValue, 1.0f);
            var isUnlocked = currentValue >= badge.RequirementValue;

            return (progress, isUnlocked);
        }

        #region Métodos de Cálculo Específicos

        private async Task<float> GetInterviewsCompletedAsync(AppDbContext db, int userId)
        {
            // Simular entrevistas completadas - en el futuro se conectará con la tabla real
            return await Task.FromResult(Random.Shared.Next(0, 15));
        }

        private async Task<float> GetGameTypeWinsAsync(AppDbContext db, int userId, string gameType)
        {
            // Simular victorias por tipo de juego
            return await Task.FromResult(Random.Shared.Next(0, 25));
        }

        private async Task<float> GetGameTypeCompletedAsync(AppDbContext db, int userId, string gameType)
        {
            // Simular juegos completados por tipo
            return await Task.FromResult(Random.Shared.Next(0, 30));
        }

        private async Task<float> GetTechnicalAnswersAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(10, 60));
        }

        private async Task<float> GetSpeedRoundsHighAccuracyAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 8));
        }

        private async Task<float> GetPatternsIdentifiedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(15, 85));
        }

        private async Task<float> GetSoftSkillsInterviewsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(2, 12));
        }

        private async Task<float> GetDailyGamesPlayedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(1, 5));
        }

        private async Task<float> GetGamesRetriedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 3));
        }

        private async Task<float> GetDifferentGameTypesPlayedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(1, 8));
        }

        private async Task<float> GetQuickSessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 3));
        }

        private async Task<float> GetAccuracyStreakAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(3, 12));
        }

        private async Task<float> GetWeeklyStreakAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(2, 10));
        }

        private async Task<float> GetNightSessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 12));
        }

        private async Task<float> GetMorningSessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 8));
        }

        private async Task<float> GetMidnightSessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 20));
        }

        private async Task<float> GetAllGameTypesPlayedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(3, 15));
        }

        private async Task<float> GetPerfectInterviewsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 7));
        }

        private async Task<float> GetPerfectSessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 15));
        }

        private async Task<float> GetFastHighAccuracyInterviewsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 3));
        }

        private async Task<float> GetGameMasteryAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(2, 12));
        }

        private async Task<float> GetDifferentProfessionsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(1, 4));
        }

        private async Task<float> GetSpeedRoundsRecordAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(5, 40));
        }

        private async Task<float> GetBalancedGameplayAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(10, 60));
        }

        private float GetProfileCompletionPercentage(User user)
        {
            float completedFields = 0;
            float totalFields = 10; // Ajustar según campos requeridos

            if (!string.IsNullOrEmpty(user.Name)) completedFields++;
            if (!string.IsNullOrEmpty(user.LastName)) completedFields++;
            if (!string.IsNullOrEmpty(user.Email)) completedFields++;
            if (!string.IsNullOrEmpty(user.Country)) completedFields++;
            if (!string.IsNullOrEmpty(user.Job)) completedFields++;
            if (!string.IsNullOrEmpty(user.Profession)) completedFields++;
            if (!string.IsNullOrEmpty(user.ExperienceLevel)) completedFields++;
            if (user.Skills.Any()) completedFields++;
            if (user.Softskills.Any()) completedFields++;
            if (user.Interests.Any()) completedFields++;

            return (completedFields / totalFields) * 100f;
        }

        private async Task<float> GetLanguagesUsedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(1, 4));
        }

        private async Task<float> GetMentoringAchievementsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 8));
        }

        private async Task<float> GetBugsReportedAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 5));
        }

        private async Task<float> GetSuccessfulReferralsAsync(AppDbContext db, int userId)
        {
            return await Task.FromResult(Random.Shared.Next(0, 7));
        }

        #endregion

        /// <summary>
        /// Actualiza el nivel del usuario basado en su experiencia
        /// </summary>
        private void UpdateUserLevel(User user)
        {
            int newLevel = (user.Experience / 1000) + 1;
            if (newLevel > user.Level)
            {
                user.Level = newLevel;
                // Aquí se podría añadir lógica adicional para recompensas por subir de nivel
            }
        }

        /// <summary>
        /// Verifica si un usuario puede desbloquear una insignia específica
        /// </summary>
        public async Task<bool> CanUnlockBadgeAsync(int userId, int badgeId)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            
            var user = await db.Users.FirstOrDefaultAsync(u => u.Id == userId);
            var badge = await db.Badges.FirstOrDefaultAsync(b => b.Id == badgeId);
            
            if (user == null || badge == null) return false;

            var (_, canUnlock) = await CalculateBadgeProgressAsync(user, badge);
            return canUnlock;
        }

        /// <summary>
        /// Obtiene el progreso actual de una insignia para un usuario
        /// </summary>
        public async Task<float> GetBadgeProgressAsync(int userId, int badgeId)
        {
            using var db = await _dbFactory.CreateDbContextAsync();
            
            var user = await db.Users.FirstOrDefaultAsync(u => u.Id == userId);
            var badge = await db.Badges.FirstOrDefaultAsync(b => b.Id == badgeId);
            
            if (user == null || badge == null) return 0f;

            var (progress, _) = await CalculateBadgeProgressAsync(user, badge);
            return progress;
        }
    }
}
