namespace Ready4Hire.MVVM.Models
{
    public class Chat
    {
        int Id { get; set; }
        List<Message> Messages { get; set; }
        string ChatMode { get; set; } // practice or exam
        int questionCount { get; set; }
        int score { get; set; }
    }
}
