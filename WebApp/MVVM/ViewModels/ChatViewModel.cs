using Microsoft.EntityFrameworkCore;
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
        }

        public async Task LoadDataAsync(int chatId)
        {
            if (chatId != 0)
            {
                var chat = await _db.Chats
                    .Include(c => c.Messages)
                    .FirstOrDefaultAsync(c => c.Id == chatId);
                Messages = chat?.Messages?.ToList();
            }
            else
            {
                Messages = new List<Message>();
            }

            User = await _db.Users.FirstOrDefaultAsync();
        }
    }
}
