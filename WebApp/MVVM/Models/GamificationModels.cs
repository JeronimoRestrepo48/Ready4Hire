namespace Ready4Hire.MVVM.Models
{
    public class UserStats
    {
        public string UserId { get; set; } = "";
        public int Level { get; set; }
        public int Experience { get; set; }
        public int TotalPoints { get; set; }
        public int TotalGamesPlayed { get; set; }
        public int TotalGamesWon { get; set; }
        public int StreakDays { get; set; }
        public int Rank { get; set; }
        public Dictionary<string, int> GamesByType { get; set; } = new();
        public Dictionary<string, int> BestScores { get; set; } = new();
    }

    public class Achievement
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public string Icon { get; set; } = "";
        public int Points { get; set; }
        public bool Unlocked { get; set; }
        public DateTime? UnlockedAt { get; set; }
        public float Progress { get; set; }
    }

    public class GameInfo
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public string Type { get; set; } = "";
        public string Profession { get; set; } = "";
        public string Difficulty { get; set; } = "";
        public int DurationMinutes { get; set; }
        public int PointsReward { get; set; }
        public bool AiPowered { get; set; } = true;
        public int Stars { get; set; } = 3; // Difficulty/reward stars (1-5)
        public int AverageTimeMinutes { get; set; } = 10; // Average completion time
    }

    public class LeaderboardEntry
    {
        public int Rank { get; set; }
        public string UserId { get; set; } = "";
        public string Username { get; set; } = "";
        public int TotalPoints { get; set; }
        public int Level { get; set; }
        public int GamesWon { get; set; }
        public int AchievementsCount { get; set; }
        public string Profession { get; set; } = "";
    }

    public class Profession
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Category { get; set; } = "";
        public string Description { get; set; } = "";
        public List<string> TechnicalSkills { get; set; } = new();
        public List<string> SoftSkills { get; set; } = new();
        public List<string> CommonTools { get; set; } = new();
        public string DifficultyLevel { get; set; } = "";
        public bool RemoteFriendly { get; set; }
    }
}

