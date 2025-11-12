using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Services
{
    /// <summary>
    /// Servicio de audit logging para trazabilidad y compliance.
    /// Registra todas las acciones críticas del sistema.
    /// </summary>
    public class AuditService
    {
        private readonly IDbContextFactory<AppDbContext> _dbFactory;
        private readonly ILogger<AuditService> _logger;

        public AuditService(
            IDbContextFactory<AppDbContext> dbFactory,
            ILogger<AuditService> logger)
        {
            _dbFactory = dbFactory;
            _logger = logger;
        }

        /// <summary>
        /// Registra un evento de auditoría
        /// </summary>
        public async Task LogAsync(AuditEvent auditEvent)
        {
            try
            {
                using var db = await _dbFactory.CreateDbContextAsync();

                var auditLog = new AuditLog
                {
                    UserId = auditEvent.UserId,
                    Action = auditEvent.Action,
                    EntityType = auditEvent.EntityType,
                    EntityId = auditEvent.EntityId,
                    Changes = JsonSerializer.Serialize(auditEvent.Changes),
                    IpAddress = auditEvent.IpAddress,
                    UserAgent = auditEvent.UserAgent,
                    Timestamp = DateTime.UtcNow,
                    Severity = auditEvent.Severity
                };

                db.AuditLogs.Add(auditLog);
                await db.SaveChangesAsync();

                // También escribir a archivo JSONL para compliance
                var logLine = JsonSerializer.Serialize(new
                {
                    auditLog.UserId,
                    auditLog.Action,
                    auditLog.EntityType,
                    auditLog.EntityId,
                    auditLog.Changes,
                    auditLog.IpAddress,
                    auditLog.UserAgent,
                    auditLog.Timestamp,
                    auditLog.Severity
                });

                var logsDir = Path.Combine(Directory.GetCurrentDirectory(), "logs");
                Directory.CreateDirectory(logsDir);
                var auditFilePath = Path.Combine(logsDir, $"audit-{DateTime.UtcNow:yyyy-MM-dd}.jsonl");
                
                await File.AppendAllTextAsync(auditFilePath, logLine + Environment.NewLine);

                _logger.LogDebug("Audit log created: {Action} by {UserId}", 
                    auditEvent.Action, auditEvent.UserId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating audit log");
                // No lanzar excepción para no interrumpir el flujo principal
            }
        }

        /// <summary>
        /// Registra un evento de seguridad crítico
        /// </summary>
        public async Task LogSecurityEventAsync(string userId, string action, string? ipAddress = null)
        {
            await LogAsync(new AuditEvent
            {
                UserId = userId,
                Action = action,
                EntityType = "Security",
                Severity = "CRITICAL",
                IpAddress = ipAddress
            });
        }

        /// <summary>
        /// Obtiene logs de auditoría por usuario
        /// </summary>
        public async Task<List<AuditLog>> GetAuditLogsByUserAsync(string userId, int limit = 100)
        {
            using var db = await _dbFactory.CreateDbContextAsync();

            return await db.AuditLogs
                .Where(a => a.UserId == userId)
                .OrderByDescending(a => a.Timestamp)
                .Take(limit)
                .ToListAsync();
        }

        /// <summary>
        /// Obtiene logs de seguridad críticos
        /// </summary>
        public async Task<List<AuditLog>> GetSecurityLogsAsync(int limit = 100)
        {
            using var db = await _dbFactory.CreateDbContextAsync();

            return await db.AuditLogs
                .Where(a => a.EntityType == "Security" && a.Severity == "CRITICAL")
                .OrderByDescending(a => a.Timestamp)
                .Take(limit)
                .ToListAsync();
        }
    }

    /// <summary>
    /// Evento de auditoría
    /// </summary>
    public class AuditEvent
    {
        public string UserId { get; set; } = string.Empty;
        public string Action { get; set; } = string.Empty;
        public string EntityType { get; set; } = string.Empty;
        public int? EntityId { get; set; }
        public object? Changes { get; set; }
        public string? IpAddress { get; set; }
        public string? UserAgent { get; set; }
        public string Severity { get; set; } = "INFO";
    }

    /// <summary>
    /// Modelo de log de auditoría
    /// </summary>
    public class AuditLog
    {
        public int Id { get; set; }
        public string UserId { get; set; } = string.Empty;
        public string Action { get; set; } = string.Empty;
        public string EntityType { get; set; } = string.Empty;
        public int? EntityId { get; set; }
        public string? Changes { get; set; } // JSON
        public string? IpAddress { get; set; }
        public string? UserAgent { get; set; }
        public DateTime Timestamp { get; set; }
        public string Severity { get; set; } = "INFO";
    }
}

