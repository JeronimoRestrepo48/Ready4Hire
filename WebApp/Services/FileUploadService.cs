using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Components.Forms;

namespace Ready4Hire.Services
{
    public class FileUploadService
    {
        private static readonly string[] AllowedExtensions = { ".pdf", ".doc", ".docx" };
        private static readonly string[] AllowedMimeTypes = 
        { 
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        };
        
        // Magic bytes para validación de tipo de archivo real
        private static readonly Dictionary<string, byte[][]> FileMagicBytes = new()
        {
            { ".pdf", new[] { new byte[] { 0x25, 0x50, 0x44, 0x46 } } }, // %PDF
            { ".doc", new[] { new byte[] { 0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1 } } }, // MS Office
            { ".docx", new[] { new byte[] { 0x50, 0x4B, 0x03, 0x04 } } } // ZIP (DOCX es un ZIP)
        };
        
        private const long MaxFileSize = 5 * 1024 * 1024; // 5 MB
        private readonly string _uploadPath;
        private readonly ILogger<FileUploadService> _logger;

        public FileUploadService(ILogger<FileUploadService> logger)
        {
            _logger = logger;
            _uploadPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads", "resumes");
            
            // Crear directorio si no existe
            if (!Directory.Exists(_uploadPath))
            {
                Directory.CreateDirectory(_uploadPath);
            }
        }

        public async Task<(bool Success, string? FilePath, string? ErrorMessage)> UploadResumeAsync(
            IBrowserFile file, 
            string userId)
        {
            try
            {
                // 1. Validación de tamaño
                if (file.Size > MaxFileSize)
                {
                    return (false, null, $"El archivo excede el tamaño máximo permitido de {MaxFileSize / 1024 / 1024} MB");
                }

                if (file.Size == 0)
                {
                    return (false, null, "El archivo está vacío");
                }

                // 2. Validación de extensión
                var extension = Path.GetExtension(file.Name).ToLowerInvariant();
                if (!AllowedExtensions.Contains(extension))
                {
                    return (false, null, $"Formato de archivo no permitido. Solo se aceptan: {string.Join(", ", AllowedExtensions)}");
                }

                // 3. Validación de tipo MIME
                if (!AllowedMimeTypes.Contains(file.ContentType))
                {
                    _logger.LogWarning($"Intento de subir archivo con MIME type no permitido: {file.ContentType}");
                    return (false, null, "Tipo de archivo no válido");
                }

                // 4. Leer el archivo para validaciones adicionales
                using var ms = new MemoryStream();
                await file.OpenReadStream(MaxFileSize).CopyToAsync(ms);
                var fileBytes = ms.ToArray();

                // 5. Validación de Magic Bytes (firma del archivo)
                if (!ValidateFileMagicBytes(fileBytes, extension))
                {
                    _logger.LogWarning($"Archivo rechazado: Magic bytes no coinciden con extensión {extension}");
                    return (false, null, "El contenido del archivo no coincide con su extensión");
                }

                // 6. Validación de contenido peligroso
                if (ContainsDangerousPatterns(fileBytes))
                {
                    _logger.LogWarning($"Archivo rechazado: Contiene patrones peligrosos");
                    return (false, null, "El archivo contiene contenido no permitido");
                }

                // 7. Generar nombre de archivo seguro y único
                var safeFileName = GenerateSafeFileName(userId, extension);
                var fullPath = Path.Combine(_uploadPath, safeFileName);

                // 8. Eliminar archivo anterior si existe
                await DeleteUserPreviousResumeAsync(userId);

                // 9. Guardar archivo de forma segura
                await File.WriteAllBytesAsync(fullPath, fileBytes);

                // 10. Validación post-guardado
                var savedFileInfo = new FileInfo(fullPath);
                if (savedFileInfo.Length != file.Size)
                {
                    File.Delete(fullPath);
                    return (false, null, "Error al guardar el archivo. Por favor, intente de nuevo.");
                }

                _logger.LogInformation($"Resume subido exitosamente: {safeFileName} para usuario {userId}");
                return (true, $"/uploads/resumes/{safeFileName}", null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error al subir resume para usuario {userId}");
                return (false, null, "Error al procesar el archivo. Por favor, intente de nuevo.");
            }
        }

        private bool ValidateFileMagicBytes(byte[] fileBytes, string extension)
        {
            if (!FileMagicBytes.ContainsKey(extension))
                return false;

            var magicBytesList = FileMagicBytes[extension];
            foreach (var magicBytes in magicBytesList)
            {
                if (fileBytes.Length >= magicBytes.Length)
                {
                    bool matches = true;
                    for (int i = 0; i < magicBytes.Length; i++)
                    {
                        if (fileBytes[i] != magicBytes[i])
                        {
                            matches = false;
                            break;
                        }
                    }
                    if (matches)
                        return true;
                }
            }
            return false;
        }

        private bool ContainsDangerousPatterns(byte[] fileBytes)
        {
            // Convertir a string para buscar patrones peligrosos
            var content = System.Text.Encoding.UTF8.GetString(fileBytes).ToLower();
            
            // Patrones peligrosos comunes
            var dangerousPatterns = new[]
            {
                "<script", "javascript:", "onerror=", "onload=",
                "<?php", "<%", "eval(", "base64_decode",
                "cmd.exe", "powershell", "/bin/bash", "/bin/sh",
                "wget ", "curl ", "nc -", "netcat",
                "exec(", "system(", "passthru(", "shell_exec"
            };

            return dangerousPatterns.Any(pattern => content.Contains(pattern));
        }

        private string GenerateSafeFileName(string userId, string extension)
        {
            // Sanitizar userId
            var safeUserId = new string(userId.Where(c => char.IsLetterOrDigit(c) || c == '-' || c == '_').ToArray());
            
            // Generar hash único basado en timestamp
            var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var uniqueId = Convert.ToBase64String(BitConverter.GetBytes(timestamp))
                .Replace("+", "")
                .Replace("/", "")
                .Replace("=", "")
                .Substring(0, 8);
            
            return $"resume_{safeUserId}_{uniqueId}{extension}";
        }

        private async Task DeleteUserPreviousResumeAsync(string userId)
        {
            try
            {
                var files = Directory.GetFiles(_uploadPath, $"resume_{userId}_*");
                foreach (var file in files)
                {
                    File.Delete(file);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, $"Error al eliminar archivos anteriores del usuario {userId}");
            }
        }

        public async Task<bool> DeleteResumeAsync(string userId)
        {
            try
            {
                await DeleteUserPreviousResumeAsync(userId);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error al eliminar resume del usuario {userId}");
                return false;
            }
        }

        public bool FileExists(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;
                
            var fullPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", filePath.TrimStart('/'));
            return File.Exists(fullPath);
        }

        public async Task<byte[]?> GetResumeAsync(string filePath)
        {
            try
            {
                if (!FileExists(filePath))
                    return null;
                    
                var fullPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", filePath.TrimStart('/'));
                return await File.ReadAllBytesAsync(fullPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error al leer archivo: {filePath}");
                return null;
            }
        }
    }
}

