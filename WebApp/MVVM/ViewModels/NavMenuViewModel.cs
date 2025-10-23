using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.MVVM.ViewModels
{
    public class NavMenuViewModel
    {
        public List<Chat>? Chats { get; set; }
        public User? CurrentUser { get; set; }
        private readonly AppDbContext _db;

        public NavMenuViewModel(AppDbContext db)
        {
            _db = db;
        }

        public async Task LoadDataAsync()
        {
            var users = await _db.Users.Include(u => u.Chats).ToListAsync();
            CurrentUser = users.FirstOrDefault();
            Chats = CurrentUser?.Chats ?? new List<Chat>();
        }
    }
}
