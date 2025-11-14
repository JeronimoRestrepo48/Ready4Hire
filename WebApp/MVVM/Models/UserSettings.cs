namespace Ready4Hire.MVVM.Models
{
    /// <summary>
    /// Modelo para almacenar las preferencias de configuración del usuario
    /// </summary>
    public class UserSettings
    {
        public int Id { get; set; }
        public int UserId { get; set; }
        
        // Preferencias de notificaciones
        public bool EmailNotifications { get; set; } = true;
        public bool AchievementNotifications { get; set; } = true;
        public bool InterviewReminders { get; set; } = true;
        
        // Preferencias de privacidad
        public bool ShowProfilePublic { get; set; } = false;
        public bool ShowStatsPublic { get; set; } = false;
        public bool AllowDataSharing { get; set; } = false;
        
        // Preferencias de idioma y región
        public string Language { get; set; } = "es";
        public string Region { get; set; } = "ES";
        public string TimeZone { get; set; } = "Europe/Madrid";
        
        // Preferencias de entrevistas
        public string DefaultDifficulty { get; set; } = "mid";
        public bool AutomaticFeedback { get; set; } = true;
        
        // Preferencias de gamificación
        public bool ShowDetailedStats { get; set; } = true;
        public bool CompetitiveMode { get; set; } = false;
        
        // Relación
        public User User { get; set; } = null!;
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }
}

