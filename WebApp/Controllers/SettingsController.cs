using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System.ComponentModel.DataAnnotations;

namespace Ready4Hire.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SettingsController : ControllerBase
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly SecurityService _securityService;
        private readonly ILogger<SettingsController> _logger;

        public SettingsController(
            IDbContextFactory<AppDbContext> dbFactory,
            SecurityService securityService,
            ILogger<SettingsController> logger)
        {
            _dbFactory = dbFactory;
            _securityService = securityService;
            _logger = logger;
        }

        /// <summary>
        /// Cambia la contraseña del usuario
        /// </summary>
        [HttpPost("change-password")]
        public async Task<IActionResult> ChangePassword([FromBody] ChangePasswordRequest request)
        {
            try
            {
                if (!_securityService.ValidateEmail(request.Email))
                {
                    return BadRequest(new { message = "Email inválido" });
                }

                var (isPasswordValid, passwordMessage) = _securityService.ValidatePassword(request.NewPassword);
                if (!isPasswordValid)
                {
                    return BadRequest(new { message = passwordMessage });
                }

                var email = request.Email.Trim().ToLower();

                using var db = await _dbFactory.CreateDbContextAsync();
                var user = await db.Users.FirstOrDefaultAsync(u => u.Email == email);

                if (user == null)
                {
                    return NotFound(new { message = "Usuario no encontrado" });
                }

                // Verificar contraseña actual
                if (!BCrypt.Net.BCrypt.Verify(request.CurrentPassword, user.Password))
                {
                    return Unauthorized(new { message = "Contraseña actual incorrecta" });
                }

                // Actualizar contraseña
                user.Password = BCrypt.Net.BCrypt.HashPassword(request.NewPassword);
                await db.SaveChangesAsync();

                _logger.LogInformation($"[SETTINGS] Password changed for user: {email}");

                return Ok(new { success = true, message = "Contraseña actualizada exitosamente" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Change password failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al cambiar la contraseña" });
            }
        }

        /// <summary>
        /// Obtiene o actualiza las preferencias de notificaciones
        /// </summary>
        [HttpGet("notifications/{userId}")]
        public async Task<IActionResult> GetNotifications(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, userId);

                return Ok(new
                {
                    emailNotifications = settings.EmailNotifications,
                    achievementNotifications = settings.AchievementNotifications,
                    interviewReminders = settings.InterviewReminders
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Get notifications failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener preferencias de notificaciones" });
            }
        }

        [HttpPost("notifications")]
        public async Task<IActionResult> UpdateNotifications([FromBody] UpdateNotificationsRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, request.UserId);

                settings.EmailNotifications = request.EmailNotifications;
                settings.AchievementNotifications = request.AchievementNotifications;
                settings.InterviewReminders = request.InterviewReminders;
                settings.UpdatedAt = DateTime.UtcNow;

                await db.SaveChangesAsync();

                _logger.LogInformation($"[SETTINGS] Notifications updated for user: {request.UserId}");

                return Ok(new { success = true, message = "Preferencias de notificaciones actualizadas" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Update notifications failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al actualizar preferencias de notificaciones" });
            }
        }

        /// <summary>
        /// Obtiene o actualiza las preferencias de privacidad y seguridad
        /// </summary>
        [HttpGet("privacy/{userId}")]
        public async Task<IActionResult> GetPrivacy(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, userId);

                return Ok(new
                {
                    showProfilePublic = settings.ShowProfilePublic,
                    showStatsPublic = settings.ShowStatsPublic,
                    allowDataSharing = settings.AllowDataSharing
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Get privacy failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener preferencias de privacidad" });
            }
        }

        [HttpPost("privacy")]
        public async Task<IActionResult> UpdatePrivacy([FromBody] UpdatePrivacyRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, request.UserId);

                settings.ShowProfilePublic = request.ShowProfilePublic;
                settings.ShowStatsPublic = request.ShowStatsPublic;
                settings.AllowDataSharing = request.AllowDataSharing;
                settings.UpdatedAt = DateTime.UtcNow;

                await db.SaveChangesAsync();

                _logger.LogInformation($"[SETTINGS] Privacy updated for user: {request.UserId}");

                return Ok(new { success = true, message = "Preferencias de privacidad actualizadas" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Update privacy failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al actualizar preferencias de privacidad" });
            }
        }

        /// <summary>
        /// Obtiene o actualiza las preferencias de idioma y región
        /// </summary>
        [HttpGet("language/{userId}")]
        public async Task<IActionResult> GetLanguage(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, userId);

                return Ok(new
                {
                    language = settings.Language,
                    region = settings.Region,
                    timeZone = settings.TimeZone
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Get language failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener preferencias de idioma" });
            }
        }

        [HttpPost("language")]
        public async Task<IActionResult> UpdateLanguage([FromBody] UpdateLanguageRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, request.UserId);

                settings.Language = request.Language;
                settings.Region = request.Region;
                settings.TimeZone = request.TimeZone ?? settings.TimeZone;
                settings.UpdatedAt = DateTime.UtcNow;

                await db.SaveChangesAsync();

                _logger.LogInformation($"[SETTINGS] Language updated for user: {request.UserId}");

                return Ok(new { success = true, message = "Preferencias de idioma actualizadas" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Update language failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al actualizar preferencias de idioma" });
            }
        }

        /// <summary>
        /// Elimina todos los datos del usuario (excepto la cuenta)
        /// </summary>
        [HttpDelete("delete-all-data/{userId}")]
        public async Task<IActionResult> DeleteAllData(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var user = await db.Users.FindAsync(userId);

                if (user == null)
                {
                    return NotFound(new { message = "Usuario no encontrado" });
                }

                // Eliminar chats y mensajes
                var chats = await db.Chats.Where(c => c.UserId == userId).ToListAsync();
                foreach (var chat in chats)
                {
                    var messages = await db.Messages.Where(m => m.ChatId == chat.Id).ToListAsync();
                    db.Messages.RemoveRange(messages);
                }
                db.Chats.RemoveRange(chats);

                // Eliminar badges del usuario
                var userBadges = await db.UserBadges.Where(ub => ub.UserId == userId).ToListAsync();
                db.UserBadges.RemoveRange(userBadges);

                // Eliminar entrevistas y reportes relacionados
                var interviews = await db.Interviews.Where(i => i.UserId == userId).ToListAsync();
                db.Interviews.RemoveRange(interviews);

                // Resetear estadísticas de gamificación
                user.Level = 1;
                user.Experience = 0;
                user.TotalPoints = 0;
                user.StreakDays = 0;
                user.TotalGamesPlayed = 0;
                user.TotalGamesWon = 0;

                await db.SaveChangesAsync();

                _logger.LogWarning($"[SETTINGS] All data deleted for user: {userId}");

                return Ok(new { success = true, message = "Todos los datos han sido eliminados" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Delete all data failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al eliminar datos" });
            }
        }

        /// <summary>
        /// Elimina la cuenta del usuario permanentemente
        /// </summary>
        [HttpDelete("delete-account/{userId}")]
        public async Task<IActionResult> DeleteAccount(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var user = await db.Users.FindAsync(userId);

                if (user == null)
                {
                    return NotFound(new { message = "Usuario no encontrado" });
                }

                // Primero eliminar todos los datos relacionados
                await DeleteAllUserData(db, userId);

                // Eliminar configuración del usuario
                var settings = await db.UserSettings.FirstOrDefaultAsync(s => s.UserId == userId);
                if (settings != null)
                {
                    db.UserSettings.Remove(settings);
                }

                // Finalmente eliminar el usuario
                db.Users.Remove(user);
                await db.SaveChangesAsync();

                _logger.LogWarning($"[SETTINGS] Account deleted permanently for user: {userId}");

                return Ok(new { success = true, message = "Cuenta eliminada permanentemente" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Delete account failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al eliminar la cuenta" });
            }
        }

        /// <summary>
        /// Obtiene todas las preferencias del usuario
        /// </summary>
        [HttpGet("all/{userId}")]
        public async Task<IActionResult> GetAllSettings(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var settings = await GetOrCreateUserSettings(db, userId);

                return Ok(new
                {
                    notifications = new
                    {
                        emailNotifications = settings.EmailNotifications,
                        achievementNotifications = settings.AchievementNotifications,
                        interviewReminders = settings.InterviewReminders
                    },
                    privacy = new
                    {
                        showProfilePublic = settings.ShowProfilePublic,
                        showStatsPublic = settings.ShowStatsPublic,
                        allowDataSharing = settings.AllowDataSharing
                    },
                    language = new
                    {
                        language = settings.Language,
                        region = settings.Region,
                        timeZone = settings.TimeZone
                    },
                    interviews = new
                    {
                        defaultDifficulty = settings.DefaultDifficulty,
                        automaticFeedback = settings.AutomaticFeedback
                    },
                    gamification = new
                    {
                        showDetailedStats = settings.ShowDetailedStats,
                        competitiveMode = settings.CompetitiveMode
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[SETTINGS ERROR] Get all settings failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener preferencias" });
            }
        }

        // Métodos auxiliares
        private async Task<UserSettings> GetOrCreateUserSettings(AppDbContext db, int userId)
        {
            var settings = await db.UserSettings.FirstOrDefaultAsync(s => s.UserId == userId);
            
            if (settings == null)
            {
                // Verificar que el usuario existe
                var user = await db.Users.FindAsync(userId);
                if (user == null)
                {
                    throw new Exception("Usuario no encontrado");
                }

                settings = new UserSettings
                {
                    UserId = userId,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };
                db.UserSettings.Add(settings);
                await db.SaveChangesAsync();
            }

            return settings;
        }

        private async Task DeleteAllUserData(AppDbContext db, int userId)
        {
            // Eliminar chats y mensajes
            var chats = await db.Chats.Where(c => c.UserId == userId).ToListAsync();
            foreach (var chat in chats)
            {
                var messages = await db.Messages.Where(m => m.ChatId == chat.Id).ToListAsync();
                db.Messages.RemoveRange(messages);
            }
            db.Chats.RemoveRange(chats);

            // Eliminar badges del usuario
            var userBadges = await db.UserBadges.Where(ub => ub.UserId == userId).ToListAsync();
            db.UserBadges.RemoveRange(userBadges);

            // Eliminar entrevistas y reportes relacionados
            var interviews = await db.Interviews.Where(i => i.UserId == userId).ToListAsync();
            db.Interviews.RemoveRange(interviews);
        }
    }

    // DTOs
    public class ChangePasswordRequest
    {
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        [Required]
        public string CurrentPassword { get; set; } = string.Empty;

        [Required]
        [MinLength(8)]
        public string NewPassword { get; set; } = string.Empty;
    }

    public class UpdateNotificationsRequest
    {
        [Required]
        public int UserId { get; set; }

        public bool EmailNotifications { get; set; } = true;
        public bool AchievementNotifications { get; set; } = true;
        public bool InterviewReminders { get; set; } = true;
    }

    public class UpdatePrivacyRequest
    {
        [Required]
        public int UserId { get; set; }

        public bool ShowProfilePublic { get; set; } = false;
        public bool ShowStatsPublic { get; set; } = false;
        public bool AllowDataSharing { get; set; } = false;
    }

    public class UpdateLanguageRequest
    {
        [Required]
        public int UserId { get; set; }

        [Required]
        public string Language { get; set; } = "es";

        [Required]
        public string Region { get; set; } = "ES";

        public string? TimeZone { get; set; }
    }
}

