using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System.ComponentModel.DataAnnotations;

namespace Ready4Hire.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly SecurityService _securityService;
        private readonly ILogger<AuthController> _logger;

        public AuthController(
            IDbContextFactory<AppDbContext> dbFactory,
            SecurityService securityService,
            ILogger<AuthController> logger)
        {
            _dbFactory = dbFactory;
            _securityService = securityService;
            _logger = logger;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register([FromBody] RegisterRequest request)
        {
            try
            {
                _logger.LogInformation($"[AUTH] Registration attempt for email: {request.Email}");

                // Validar email
                if (!_securityService.ValidateEmail(request.Email))
                {
                    return BadRequest(new { message = "Email inválido" });
                }

                // Validar contraseña
                var (isPasswordValid, passwordMessage) = _securityService.ValidatePassword(request.Password);
                if (!isPasswordValid)
                {
                    return BadRequest(new { message = passwordMessage });
                }

                // Sanitizar inputs
                var email = _securityService.SanitizeInput(request.Email.Trim().ToLower());
                var name = _securityService.SanitizeInput(request.Name);
                var lastName = _securityService.SanitizeInput(request.LastName);
                var country = _securityService.SanitizeInput(request.Country);
                var job = _securityService.SanitizeInput(request.Job);

                // Validar longitud
                if (!_securityService.ValidateLength(name, 100) ||
                    !_securityService.ValidateLength(lastName, 100) ||
                    !_securityService.ValidateLength(job, 200))
                {
                    return BadRequest(new { message = "Uno o más campos exceden el límite de caracteres" });
                }

                using var db = await _dbFactory.CreateDbContextAsync();

                // Verificar si el usuario ya existe
                var existingUser = await db.Users.FirstOrDefaultAsync(u => u.Email == email);
                if (existingUser != null)
                {
                    return Conflict(new { message = "El correo electrónico ya está registrado" });
                }

                // Hash de contraseña
                var hashedPassword = BCrypt.Net.BCrypt.HashPassword(request.Password);

                // Crear usuario
                var user = new User
                {
                    Email = email,
                    Password = hashedPassword,
                    Name = name,
                    LastName = lastName,
                    Country = country,
                    Job = job,
                    Skills = request.Skills ?? new List<string>(),
                    Softskills = request.Softskills ?? new List<string>(),
                    Interests = request.Interests ?? new List<string>()
                };

                db.Users.Add(user);
                await db.SaveChangesAsync();

                _logger.LogInformation($"[AUTH] User registered successfully: {email} (ID: {user.Id})");

                return Ok(new
                {
                    success = true,
                    message = "Usuario registrado exitosamente",
                    userId = user.Id,
                    email = user.Email,
                    name = user.Name
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[AUTH ERROR] Registration failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al registrar el usuario. Por favor intenta nuevamente." });
            }
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login([FromBody] LoginRequest request)
        {
            try
            {
                _logger.LogInformation($"[AUTH] Login attempt for email: {request.Email}");

                if (!_securityService.ValidateEmail(request.Email))
                {
                    return BadRequest(new { message = "Email inválido" });
                }

                var email = request.Email.Trim().ToLower();

                using var db = await _dbFactory.CreateDbContextAsync();

                var user = await db.Users.FirstOrDefaultAsync(u => u.Email == email);
                if (user == null)
                {
                    _logger.LogWarning($"[AUTH] Login failed: User not found - {email}");
                    return Unauthorized(new { message = "Credenciales incorrectas" });
                }

                // Verificar contraseña
                if (!BCrypt.Net.BCrypt.Verify(request.Password, user.Password))
                {
                    _logger.LogWarning($"[AUTH] Login failed: Invalid password - {email}");
                    return Unauthorized(new { message = "Credenciales incorrectas" });
                }

                _logger.LogInformation($"[AUTH] Login successful: {email} (ID: {user.Id})");

                return Ok(new
                {
                    success = true,
                    message = "Login exitoso",
                    user = new
                    {
                        id = user.Id,
                        email = user.Email,
                        name = user.Name,
                        lastName = user.LastName,
                        country = user.Country,
                        job = user.Job,
                        skills = user.Skills,
                        softskills = user.Softskills,
                        interests = user.Interests
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[AUTH ERROR] Login failed: {ex.Message}");
                return StatusCode(500, new { message = "Error al iniciar sesión. Por favor intenta nuevamente." });
            }
        }

        [HttpPost("validate-session")]
        public async Task<IActionResult> ValidateSession([FromBody] ValidateSessionRequest request)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();
                var user = await db.Users.FindAsync(request.UserId);

                if (user == null)
                {
                    return Unauthorized(new { valid = false, message = "Sesión inválida" });
                }

                return Ok(new { valid = true, user = new { id = user.Id, email = user.Email, name = user.Name } });
            }
            catch (Exception ex)
            {
                _logger.LogError($"[AUTH ERROR] Session validation failed: {ex.Message}");
                return StatusCode(500, new { valid = false, message = "Error al validar sesión" });
            }
        }
    }

    // DTOs
    public class RegisterRequest
    {
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        [Required]
        [MinLength(8)]
        public string Password { get; set; } = string.Empty;

        [Required]
        public string Name { get; set; } = string.Empty;

        [Required]
        public string LastName { get; set; } = string.Empty;

        [Required]
        public string Country { get; set; } = string.Empty;

        [Required]
        public string Job { get; set; } = string.Empty;

        public List<string>? Skills { get; set; }
        public List<string>? Softskills { get; set; }
        public List<string>? Interests { get; set; }
    }

    public class LoginRequest
    {
        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;

        [Required]
        public string Password { get; set; } = string.Empty;
    }

    public class ValidateSessionRequest
    {
        [Required]
        public int UserId { get; set; }
    }
}

