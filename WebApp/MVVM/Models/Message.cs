namespace Ready4Hire.MVVM.Models
{
    public class Message
    {
        public int Id { get; set; }
        public string Text { get; set; } = string.Empty;
        public bool IsUser { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        // Clave foránea
        public int ChatId { get; set; }
        public Chat Chat { get; set; } = null!;

    }
}
