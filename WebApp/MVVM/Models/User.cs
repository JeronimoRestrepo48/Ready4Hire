namespace Ready4Hire.MVVM.Models
{
    public class User
    {
        public int Id { get; set; }
        public string Email { get; set; }
        public string Password { get; set; }
        public string Name { get; set; }
        public string LastName { get; set; }
        public string Country { get; set; }
        public string Job { get; set; }
        public List<string> Skills { get; set; }
        public List<string> Softskills { get; set; }
        public List<string> Interests { get; set; }
        public bool isAdmin { get; set; }
    }
}
