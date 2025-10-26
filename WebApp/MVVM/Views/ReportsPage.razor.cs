using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Ready4Hire.MVVM.Views
{
    public partial class ReportsPage
    {
        // State
        private List<InterviewReport> reports = new();
        private List<InterviewReport> filteredReports = new();
        private List<string> availableRoles = new();
        private bool isLoading = true;
        
        // Filters
        private string filterRole = "";
        private string filterPeriod = "all";
        private string filterMode = "";
        
        // Computed metrics
        private double averageScore = 0.0;
        private int successRate = 0;
        private int certificatesCount = 0;
        
        protected override async Task OnInitializedAsync()
        {
            await LoadReports();
        }
        
        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender && reports.Any())
            {
                await InitializeCharts();
            }
        }
        
        private async Task LoadReports()
        {
            try
            {
                isLoading = true;
                StateHasChanged();
                
                // TODO: Implementar llamada real a API
                // Por ahora, datos de ejemplo
                reports = GenerateMockReports();
                
                availableRoles = reports.Select(r => r.Role).Distinct().ToList();
                filteredReports = reports;
                
                CalculateMetrics();
                
                isLoading = false;
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading reports: {ex.Message}");
                isLoading = false;
                StateHasChanged();
            }
        }
        
        private List<InterviewReport> GenerateMockReports()
        {
            var random = new Random();
            var roles = new[] { "Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer" };
            var modes = new[] { "practice", "exam" };
            
            var mockReports = new List<InterviewReport>();
            
            for (int i = 0; i < 10; i++)
            {
                var score = 6.0 + random.NextDouble() * 4.0; // 6.0 - 10.0
                var totalQuestions = 10;
                var correctAnswers = (int)Math.Round(score);
                
                mockReports.Add(new InterviewReport
                {
                    InterviewId = Guid.NewGuid().ToString(),
                    Role = roles[random.Next(roles.Length)],
                    Mode = modes[random.Next(modes.Length)],
                    CompletedAt = DateTime.Now.AddDays(-random.Next(30)),
                    AverageScore = score,
                    TotalQuestions = totalQuestions,
                    CorrectAnswers = correctAnswers,
                    TotalTimeSeconds = random.Next(600, 1800),
                    CertificateId = (modes[i % 2] == "exam" && score >= 7.5) ? $"R4H-{Guid.NewGuid().ToString().Substring(0, 12).ToUpper()}" : null
                });
            }
            
            return mockReports.OrderByDescending(r => r.CompletedAt).ToList();
        }
        
        private void CalculateMetrics()
        {
            if (!filteredReports.Any()) return;
            
            averageScore = filteredReports.Average(r => r.AverageScore);
            successRate = (int)Math.Round((double)filteredReports.Count(r => r.AverageScore >= 7.0) / filteredReports.Count * 100);
            certificatesCount = filteredReports.Count(r => !string.IsNullOrEmpty(r.CertificateId));
        }
        
        private void ApplyFilters()
        {
            filteredReports = reports.Where(r =>
            {
                // Filter by role
                if (!string.IsNullOrEmpty(filterRole) && r.Role != filterRole)
                    return false;
                
                // Filter by mode
                if (!string.IsNullOrEmpty(filterMode) && r.Mode != filterMode)
                    return false;
                
                // Filter by period
                if (filterPeriod != "all")
                {
                    var cutoffDate = filterPeriod switch
                    {
                        "week" => DateTime.Now.AddDays(-7),
                        "month" => DateTime.Now.AddMonths(-1),
                        "year" => DateTime.Now.AddYears(-1),
                        _ => DateTime.MinValue
                    };
                    
                    if (r.CompletedAt < cutoffDate)
                        return false;
                }
                
                return true;
            }).ToList();
            
            CalculateMetrics();
            StateHasChanged();
        }
        
        private async Task RefreshReports()
        {
            await LoadReports();
        }
        
        private async Task ExportAllReports()
        {
            // TODO: Implementar exportación a CSV/PDF
            await JSRuntime.InvokeVoidAsync("alert", "Exportando reportes... (Funcionalidad en desarrollo)");
        }
        
        private void ViewDetailedReport(string interviewId)
        {
            Nav.NavigateTo($"/reports/{interviewId}");
        }
        
        private async Task DownloadCertificate(string certificateId)
        {
            // TODO: Implementar descarga de certificado
            await JSRuntime.InvokeVoidAsync("alert", $"Descargando certificado {certificateId}... (Funcionalidad en desarrollo)");
        }
        
        private async Task ShareReport(string interviewId)
        {
            // TODO: Implementar compartir reporte
            var shareUrl = $"{Nav.BaseUri}reports/{interviewId}";
            await JSRuntime.InvokeVoidAsync("alert", $"URL para compartir: {shareUrl}");
        }
        
        private string GetScoreClass(double score)
        {
            if (score >= 9.0) return "excellent";
            if (score >= 8.0) return "good";
            if (score >= 7.0) return "fair";
            return "poor";
        }
        
        private string FormatTime(int seconds)
        {
            var minutes = seconds / 60;
            var remainingSeconds = seconds % 60;
            return $"{minutes}m {remainingSeconds}s";
        }
        
        private async Task InitializeCharts()
        {
            try
            {
                // Preparar datos para gráfica de tendencia
                var chartData = filteredReports
                    .OrderBy(r => r.CompletedAt)
                    .Select(r => new
                    {
                        date = r.CompletedAt.ToString("dd/MM"),
                        score = r.AverageScore
                    })
                    .ToArray();
                
                await JSRuntime.InvokeVoidAsync("initializePerformanceTrendChart", chartData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing charts: {ex.Message}");
            }
        }
    }
    
    public class InterviewReport
    {
        public string InterviewId { get; set; }
        public string Role { get; set; }
        public string Mode { get; set; }
        public DateTime CompletedAt { get; set; }
        public double AverageScore { get; set; }
        public int TotalQuestions { get; set; }
        public int CorrectAnswers { get; set; }
        public int TotalTimeSeconds { get; set; }
        public string CertificateId { get; set; }
    }
}

