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

        // Relación uno-a-muchos
        public List<Chat> Chats { get; set; } = new();
    }
}
