namespace Ready4Hire.MVVM.Models
{
    public class Message
    {
        public string Text { get; set; }
        //Saber si el mensaje es del usuario o del sistema
        public bool IsUser { get; set; }
    }
}
