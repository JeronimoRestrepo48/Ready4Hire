using Microsoft.AspNetCore.Http;

namespace Ready4Hire.Middleware
{
    /// <summary>
    /// Middleware para agregar headers de seguridad OWASP recomendados.
    /// Protege contra XSS, Clickjacking, MIME sniffing y otros ataques.
    /// </summary>
    public class SecurityHeadersMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<SecurityHeadersMiddleware> _logger;

        public SecurityHeadersMiddleware(RequestDelegate next, ILogger<SecurityHeadersMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            // Headers de seguridad OWASP Top 10
            context.Response.Headers.Append("X-Content-Type-Options", "nosniff");
            context.Response.Headers.Append("X-Frame-Options", "DENY");
            context.Response.Headers.Append("X-XSS-Protection", "1; mode=block");
            context.Response.Headers.Append("Referrer-Policy", "strict-origin-when-cross-origin");

            // HSTS (HTTP Strict Transport Security) solo en HTTPS
            if (context.Request.IsHttps)
            {
                context.Response.Headers.Append("Strict-Transport-Security", 
                    "max-age=31536000; includeSubDomains; preload");
            }

            // Content Security Policy
            // Permitir scripts inline para Blazor y recursos locales
            context.Response.Headers.Append("Content-Security-Policy",
                "default-src 'self'; " +
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
                "style-src 'self' 'unsafe-inline'; " +
                "img-src 'self' data: https:; " +
                "font-src 'self' data:; " +
                "connect-src 'self' http://localhost:8001 ws: wss:; " +
                "frame-ancestors 'none';");

            // Permissions Policy (antes Feature-Policy)
            context.Response.Headers.Append("Permissions-Policy",
                "geolocation=(), " +
                "microphone=(), " +
                "camera=(), " +
                "payment=(), " +
                "usb=(), " +
                "magnetometer=(), " +
                "gyroscope=(), " +
                "speaker=()");

            // Cross-Origin Embedder Policy
            context.Response.Headers.Append("Cross-Origin-Embedder-Policy", "require-corp");

            // Cross-Origin Opener Policy
            context.Response.Headers.Append("Cross-Origin-Opener-Policy", "same-origin");

            // Cross-Origin Resource Policy
            context.Response.Headers.Append("Cross-Origin-Resource-Policy", "same-origin");

            await _next(context);
        }
    }

    /// <summary>
    /// Extension method para registrar el middleware fácilmente
    /// </summary>
    public static class SecurityHeadersMiddlewareExtensions
    {
        public static IApplicationBuilder UseSecurityHeaders(this IApplicationBuilder builder)
        {
            return builder.UseMiddleware<SecurityHeadersMiddleware>();
        }
    }
}

