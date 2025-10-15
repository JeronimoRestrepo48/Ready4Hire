using Microsoft.EntityFrameworkCore;
using Ready4Hire.Components;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;

//obtener la cadena de conexión desde las variables de entorno
DotNetEnv.Env.Load();
var connectionString = Environment.GetEnvironmentVariable("POSTGRES_CONNECTION");

var builder = WebApplication.CreateBuilder(args);

// Configurar el DbContext con PostgreSQL
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseNpgsql(connectionString));


// Registrar el servicio de consumo de la API Python
builder.Services.AddHttpClient<InterviewApiService>();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();


app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
