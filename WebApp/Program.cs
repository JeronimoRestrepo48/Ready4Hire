using Microsoft.EntityFrameworkCore;
using Ready4Hire.Components;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

//obtener la cadena de conexi�n desde las variables de entorno
DotNetEnv.Env.Load();
var connectionString = Environment.GetEnvironmentVariable("POSTGRES_CONNECTION");

var builder = WebApplication.CreateBuilder(args);

// Configurar DbContextFactory para evitar problemas de concurrencia en Blazor Server
builder.Services.AddDbContextFactory<AppDbContext>(options =>
    options.UseNpgsql(connectionString));

// Servicios de seguridad y autenticación
builder.Services.AddScoped<Ready4Hire.Services.AuthService>();
builder.Services.AddScoped<Ready4Hire.Services.SecurityService>();

// HttpClient para APIs
builder.Services.AddHttpClient();

// Registrar el servicio de consumo de la API Python
builder.Services.AddHttpClient<InterviewApiService>();

// Agregar controladores API
builder.Services.AddControllers();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// Configurar headers de seguridad
builder.Services.AddAntiforgery();

// Configurar CORS para permitir conexiones locales
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowLocalhost", policy =>
    {
        policy.WithOrigins("http://localhost:5214", "https://localhost:5214")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

// Headers de seguridad para prevenir XSS, Clickjacking, etc.
app.Use(async (context, next) =>
{
    context.Response.Headers["X-Content-Type-Options"] = "nosniff";
    context.Response.Headers["X-Frame-Options"] = "DENY";
    context.Response.Headers["X-XSS-Protection"] = "1; mode=block";
    context.Response.Headers["Referrer-Policy"] = "strict-origin-when-cross-origin";
    context.Response.Headers["Content-Security-Policy"] = 
        "default-src 'self'; " +
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
        "style-src 'self' 'unsafe-inline'; " +
        "img-src 'self' data: https:; " +
        "font-src 'self' data:; " +
        "connect-src 'self' http://localhost:8001;";
    
    await next();
});

app.UseHttpsRedirection();

app.UseCors("AllowLocalhost");

app.UseAntiforgery();

// Mapear controladores API
app.MapControllers();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
