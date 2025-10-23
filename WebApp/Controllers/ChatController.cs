using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using System.ComponentModel.DataAnnotations;

namespace Ready4Hire.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ChatController : ControllerBase
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ILogger<ChatController> _logger;

        public ChatController(
            IDbContextFactory<AppDbContext> dbFactory,
            ILogger<ChatController> logger)
        {
            _dbFactory = dbFactory;
            _logger = logger;
        }

        [HttpGet("user/{userId}")]
        public async Task<IActionResult> GetUserChats(int userId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                var chats = await db.Chats
                    .Where(c => c.UserId == userId)
                    .OrderByDescending(c => c.CreatedAt)
                    .Select(c => new
                    {
                        id = c.Id,
                        title = c.Title,
                        createdAt = c.CreatedAt
                    })
                    .ToListAsync();

                return Ok(chats);
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to get chats for user {userId}: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener los chats" });
            }
        }

        [HttpGet("{chatId}")]
        public async Task<IActionResult> GetChat(int chatId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                var chat = await db.Chats
                    .Include(c => c.Messages.OrderBy(m => m.Timestamp))
                    .FirstOrDefaultAsync(c => c.Id == chatId);

                if (chat == null)
                {
                    return NotFound(new { message = "Chat no encontrado" });
                }

                return Ok(new
                {
                    id = chat.Id,
                    title = chat.Title,
                    userId = chat.UserId,
                    createdAt = chat.CreatedAt,
                    messages = chat.Messages.Select(m => new
                    {
                        id = m.Id,
                        content = m.Text,
                        isUser = m.IsUser,
                        timestamp = m.Timestamp
                    })
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to get chat {chatId}: {ex.Message}");
                return StatusCode(500, new { message = "Error al obtener el chat" });
            }
        }

        [HttpPost]
        public async Task<IActionResult> CreateChat([FromBody] CreateChatRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                // Verificar que el usuario existe
                var userExists = await db.Users.AnyAsync(u => u.Id == request.UserId);
                if (!userExists)
                {
                    return BadRequest(new { message = "Usuario no encontrado" });
                }

                var chat = new Chat
                {
                    UserId = request.UserId,
                    Title = string.IsNullOrEmpty(request.Title) ? "Nueva Conversación" : request.Title,
                    CreatedAt = DateTime.UtcNow
                };

                db.Chats.Add(chat);
                await db.SaveChangesAsync();

                _logger.LogInformation($"[CHAT] New chat created: ID {chat.Id} for user {request.UserId}");

                return Ok(new
                {
                    id = chat.Id,
                    title = chat.Title,
                    userId = chat.UserId,
                    createdAt = chat.CreatedAt
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to create chat: {ex.Message}");
                return StatusCode(500, new { message = "Error al crear el chat" });
            }
        }

        [HttpPost("{chatId}/messages")]
        public async Task<IActionResult> AddMessage(int chatId, [FromBody] AddMessageRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                var chat = await db.Chats.FindAsync(chatId);
                if (chat == null)
                {
                    return NotFound(new { message = "Chat no encontrado" });
                }

                var message = new Message
                {
                    ChatId = chatId,
                    Text = request.Content,
                    IsUser = request.IsUser,
                    Timestamp = DateTime.UtcNow
                };

                db.Messages.Add(message);
                await db.SaveChangesAsync();

                return Ok(new
                {
                    id = message.Id,
                    content = message.Text,
                    isUser = message.IsUser,
                    timestamp = message.Timestamp
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to add message to chat {chatId}: {ex.Message}");
                return StatusCode(500, new { message = "Error al guardar el mensaje" });
            }
        }

        [HttpPut("{chatId}/title")]
        public async Task<IActionResult> UpdateChatTitle(int chatId, [FromBody] UpdateTitleRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                var chat = await db.Chats.FindAsync(chatId);
                if (chat == null)
                {
                    return NotFound(new { message = "Chat no encontrado" });
                }

                chat.Title = request.Title;
                await db.SaveChangesAsync();

                return Ok(new { success = true, title = chat.Title });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to update chat title {chatId}: {ex.Message}");
                return StatusCode(500, new { message = "Error al actualizar el título" });
            }
        }

        [HttpDelete("{chatId}")]
        public async Task<IActionResult> DeleteChat(int chatId)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                
                var chat = await db.Chats
                    .Include(c => c.Messages)
                    .FirstOrDefaultAsync(c => c.Id == chatId);

                if (chat == null)
                {
                    return NotFound(new { message = "Chat no encontrado" });
                }

                db.Chats.Remove(chat);
                await db.SaveChangesAsync();

                _logger.LogInformation($"[CHAT] Chat deleted: ID {chatId}");

                return Ok(new { success = true, message = "Chat eliminado" });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[CHAT ERROR] Failed to delete chat {chatId}: {ex.Message}");
                return StatusCode(500, new { message = "Error al eliminar el chat" });
            }
        }
    }

    // DTOs
    public class CreateChatRequest
    {
        [Required]
        public int UserId { get; set; }
        public string? Title { get; set; }
    }

    public class AddMessageRequest
    {
        [Required]
        public string Content { get; set; } = string.Empty;
        [Required]
        public bool IsUser { get; set; }
    }

    public class UpdateTitleRequest
    {
        [Required]
        public string Title { get; set; } = string.Empty;
    }
}

