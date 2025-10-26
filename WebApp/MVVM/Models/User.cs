namespace Ready4Hire.MVVM.Models
{
    public class User
    {
        public int Id { get; set; }
        public string Email { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string LastName { get; set; } = string.Empty;
        public string Country { get; set; } = string.Empty;
        public string Job { get; set; } = string.Empty;

        // Convertidos a JSON
        public List<string> Skills { get; set; } = new();
        public List<string> Softskills { get; set; } = new();
        public List<string> Interests { get; set; } = new();

        // ✨ GAMIFICACIÓN
        public string? Profession { get; set; }
        public string? ProfessionCategory { get; set; }
        public string? ExperienceLevel { get; set; } // junior, mid, senior
        public int Level { get; set; } = 1;
        public int Experience { get; set; } = 0;
        public int TotalPoints { get; set; } = 0;
        public int StreakDays { get; set; } = 0;
        public DateTime? LastActivityDate { get; set; }
        public int TotalGamesPlayed { get; set; } = 0;
        public int TotalGamesWon { get; set; } = 0;
        public string? AvatarUrl { get; set; }
        public string? Bio { get; set; }

        // Relaciones
        public List<Chat> Chats { get; set; } = new();
        public List<UserBadge> Badges { get; set; } = new();
    }

    // ============================================================================
    // BADGES E INSIGNIAS
    // ============================================================================

    public class Badge
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Icon { get; set; } = "🏆";
        public string Category { get; set; } = "general"; // general, technical, soft_skills, achievement, milestone
        public string Rarity { get; set; } = "common"; // common, rare, epic, legendary
        public int PointsRequired { get; set; } = 0;
        public string? RequirementType { get; set; } // interviews_completed, games_won, streak_days, etc.
        public int RequirementValue { get; set; } = 0;
        public int RewardPoints { get; set; } = 0;
        public int RewardXp { get; set; } = 0;
        public bool IsActive { get; set; } = true;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // Relación
        public List<UserBadge> UserBadges { get; set; } = new();
    }

    public class UserBadge
    {
        public int Id { get; set; }
        public int UserId { get; set; }
        public int BadgeId { get; set; }
        public DateTime UnlockedAt { get; set; } = DateTime.UtcNow;
        public float Progress { get; set; } = 0;
        public bool IsUnlocked { get; set; } = false;

        // Navegación
        public User User { get; set; } = null!;
        public Badge Badge { get; set; } = null!;
    }
}
