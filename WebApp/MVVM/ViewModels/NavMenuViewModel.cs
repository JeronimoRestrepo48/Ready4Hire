using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.MVVM.ViewModels
{
    public class NavMenuViewModel
    {
        public List<Chat> Chats { get; set; }
        public User CurrentUser { get; set; }
        private readonly AppDbContext _db;

        public NavMenuViewModel(AppDbContext db)
        {
            _db = db;
            GetChats();
        }

        private async void GetChats()
        {
            var users = await _db.Users.ToListAsync();

            CurrentUser = users.FirstOrDefault();

            Chats = CurrentUser.Chats;
        }
    }
}
