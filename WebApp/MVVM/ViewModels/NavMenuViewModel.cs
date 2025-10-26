using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;

namespace Ready4Hire.MVVM.ViewModels
{
    public class NavMenuViewModel
    {
        public List<Chat>? Chats { get; set; }
        public User? CurrentUser { get; set; }
        private readonly AppDbContext _db;
        private readonly AuthService? _authService;

        public NavMenuViewModel(AppDbContext db, AuthService? authService = null)
        {
            _db = db;
            _authService = authService;
        }

        public async Task LoadDataAsync()
        {
            // Intentar obtener el usuario autenticado actual
            if (_authService != null)
            {
                var authenticatedUser = await _authService.GetCurrentUserAsync();
                if (authenticatedUser != null && !string.IsNullOrEmpty(authenticatedUser.Email))
                {
                    // Recargar el usuario desde la DB con sus chats incluidos
                    CurrentUser = await _db.Users
                        .Include(u => u.Chats)
                        .FirstOrDefaultAsync(u => u.Email == authenticatedUser.Email);
                }
            }
            
            // Fallback: obtener el primer usuario (solo para compatibilidad)
            if (CurrentUser == null)
            {
                var users = await _db.Users.Include(u => u.Chats).ToListAsync();
                CurrentUser = users.FirstOrDefault();
            }
            
            Chats = CurrentUser?.Chats ?? new List<Chat>();
        }
    }
}
