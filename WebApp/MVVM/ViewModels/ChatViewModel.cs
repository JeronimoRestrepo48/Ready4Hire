using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.MVVM.ViewModels
{
    public class ChatViewModel
    {
        private readonly AppDbContext _db;
        public List<Message>? Messages { get; set; }
        public User? User { get; set; }
        public ChatViewModel(AppDbContext db, int chatid)
        {
            _db = db;

            if (chatid != 0)
                Messages = db.Chats.Find(chatid)?.Messages?.ToList();
            User = db.Users.FirstOrDefault();
        }
    }
}
