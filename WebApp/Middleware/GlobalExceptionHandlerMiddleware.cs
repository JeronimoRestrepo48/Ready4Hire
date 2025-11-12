using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Net;
using System.Text.Json;

namespace Ready4Hire.Middleware
{
    /// <summary>
    /// Middleware global para manejar todas las excepciones no capturadas.
    /// Retorna respuestas consistentes y loguea errores apropiadamente.
    /// </summary>
    public class GlobalExceptionHandlerMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<GlobalExceptionHandlerMiddleware> _logger;
        private readonly IWebHostEnvironment _environment;

        public GlobalExceptionHandlerMiddleware(
            RequestDelegate next,
            ILogger<GlobalExceptionHandlerMiddleware> logger,
            IWebHostEnvironment environment)
        {
            _next = next;
            _logger = logger;
            _environment = environment;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (Exception ex)
            {
                await HandleExceptionAsync(context, ex);
            }
        }

        private async Task HandleExceptionAsync(HttpContext context, Exception exception)
        {
            context.Response.ContentType = "application/json";
            var response = context.Response;

            var errorResponse = new ErrorResponse
            {
                Timestamp = DateTime.UtcNow,
                Path = context.Request.Path,
                Method = context.Request.Method
            };

            // Handle specific exception types
            if (exception is DbUpdateConcurrencyException dbEx)
            {
                _logger.LogWarning(dbEx, "Concurrency conflict on {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.Conflict;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "The record has been modified by another user. Please refresh and try again.";
                errorResponse.Details = _environment.IsDevelopment() ? dbEx.Message : null;
            }
            else if (exception is DbUpdateException dbUpEx)
            {
                _logger.LogError(dbUpEx, "Database update error on {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.BadRequest;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "An error occurred while saving data to the database.";
                errorResponse.Details = _environment.IsDevelopment() ? dbUpEx.Message : null;
            }
            else if (exception is UnauthorizedAccessException unAuthEx)
            {
                _logger.LogWarning(unAuthEx, "Unauthorized access attempt to {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.Unauthorized;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "You are not authorized to access this resource.";
                errorResponse.Details = _environment.IsDevelopment() ? unAuthEx.Message : null;
            }
            else if (exception is ArgumentException argEx)
            {
                _logger.LogWarning(argEx, "Validation error on {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.BadRequest;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = argEx.Message;
                errorResponse.Details = _environment.IsDevelopment() ? argEx.ToString() : null;
            }
            else if (exception is KeyNotFoundException keyEx)
            {
                _logger.LogWarning(keyEx, "Resource not found: {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.NotFound;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "The requested resource was not found.";
                errorResponse.Details = _environment.IsDevelopment() ? keyEx.Message : null;
            }
            else if (exception is TimeoutException timeoutEx)
            {
                _logger.LogError(timeoutEx, "Timeout error on {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.RequestTimeout;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "The request timed out. Please try again.";
                errorResponse.Details = _environment.IsDevelopment() ? timeoutEx.Message : null;
            }
            else
            {
                _logger.LogError(exception, "Unhandled exception on {Path}", context.Request.Path);
                response.StatusCode = (int)HttpStatusCode.InternalServerError;
                errorResponse.StatusCode = response.StatusCode;
                errorResponse.Message = "An internal server error occurred.";
                errorResponse.Details = _environment.IsDevelopment() ? exception.ToString() : null;
            }

            // Add request ID for correlation
            errorResponse.RequestId = context.TraceIdentifier;

            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            var json = JsonSerializer.Serialize(errorResponse, options);
            await response.WriteAsync(json);
        }

        /// <summary>
        /// Modelo de respuesta de error estándar
        /// </summary>
        public class ErrorResponse
        {
            public DateTime Timestamp { get; set; }
            public int StatusCode { get; set; }
            public string Message { get; set; } = string.Empty;
            public string? Details { get; set; }
            public string Path { get; set; } = string.Empty;
            public string Method { get; set; } = string.Empty;
            public string RequestId { get; set; } = string.Empty;
        }
    }

    /// <summary>
    /// Extension method para registrar el middleware fácilmente
    /// </summary>
    public static class GlobalExceptionHandlerMiddlewareExtensions
    {
        public static IApplicationBuilder UseGlobalExceptionHandler(this IApplicationBuilder builder)
        {
            return builder.UseMiddleware<GlobalExceptionHandlerMiddleware>();
        }
    }
}

