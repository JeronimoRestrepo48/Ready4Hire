namespace Ready4Hire.MVVM.Models
{
    public class Chat
    {
        public int Id { get; set; }
        public string Title { get; set; } = string.Empty;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public string ChatMode { get; set; } = string.Empty;
        public int QuestionCount { get; set; }
        public int Score { get; set; }

        // Relación uno-a-muchos
        public List<Message> Messages { get; set; } = new();

        // Clave foránea
        public int UserId { get; set; }
        public User User { get; set; } = null!;

    }
}
