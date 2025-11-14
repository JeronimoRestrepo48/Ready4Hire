using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace Ready4Hire.MVVM.Views
{
    public partial class ReportsPage
    {
        [Inject] private InterviewApiService InterviewApi { get; set; }
        [Inject] private AuthService AuthService { get; set; }
        [Inject] private IJSRuntime JSRuntime { get; set; }
        [Inject] private NavigationManager Nav { get; set; }
        
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
                
                // Obtener email del usuario actual
                var email = await AuthService.GetCurrentUserEmailAsync();
                if (string.IsNullOrEmpty(email))
                {
                    reports = new List<InterviewReport>();
                    filteredReports = new List<InterviewReport>();
                    isLoading = false;
                    StateHasChanged();
                    return;
                }
                
                // Convertir email a user_id (formato del backend)
                var userId = $"user_{email.Replace("@", "_at_").Replace(".", "_")}";
                
                // Llamar a la API para obtener entrevistas completadas
                var response = await InterviewApi.GetCompletedInterviewsAsync(userId, 50);
                
                reports = new List<InterviewReport>();
                availableRoles = new List<string>();
                
                if (response.TryGetProperty("interviews", out var interviewsElement))
                {
                    foreach (var interview in interviewsElement.EnumerateArray())
                    {
                        var avgScore = interview.TryGetProperty("average_score", out var score) ? score.GetDouble() : 0.0;
                        var totalQuestions = interview.TryGetProperty("total_questions", out var total) ? total.GetInt32() : 0;
                        var correctAnswers = totalQuestions > 0 ? (int)Math.Round(avgScore / 10.0 * totalQuestions) : 0;
                        
                        var report = new InterviewReport
                        {
                            InterviewId = interview.TryGetProperty("interview_id", out var id) ? id.GetString() : "",
                            Role = interview.TryGetProperty("role", out var role) ? role.GetString() : "",
                            Mode = interview.TryGetProperty("mode", out var mode) ? mode.GetString() : "",
                            AverageScore = avgScore,
                            TotalQuestions = totalQuestions,
                            CorrectAnswers = correctAnswers,
                            TotalTimeSeconds = 0, // No disponible en la respuesta actual
                            CertificateId = interview.TryGetProperty("certificate_id", out var certId) ? certId.GetString() : null
                        };
                        
                        if (interview.TryGetProperty("completed_at", out var completedAt) && completedAt.ValueKind == JsonValueKind.String)
                        {
                            if (DateTime.TryParse(completedAt.GetString(), out var date))
                            {
                                report.CompletedAt = date;
                            }
                            else
                            {
                                report.CompletedAt = DateTime.Now;
                            }
                        }
                        else
                        {
                            report.CompletedAt = DateTime.Now;
                        }
                        
                        reports.Add(report);
                        
                        // Agregar rol a la lista de roles disponibles
                        if (!string.IsNullOrEmpty(report.Role) && !availableRoles.Contains(report.Role))
                        {
                            availableRoles.Add(report.Role);
                        }
                    }
                }
                
                filteredReports = reports.ToList();
                CalculateMetrics();
                
                isLoading = false;
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading reports: {ex.Message}");
                reports = new List<InterviewReport>();
                filteredReports = new List<InterviewReport>();
                isLoading = false;
                StateHasChanged();
            }
        }
        
        // Método eliminado - ya no generamos datos mock
        
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
            try
            {
                // Buscar la entrevista asociada al certificado
                var report = reports.FirstOrDefault(r => r.CertificateId == certificateId);
                if (report == null)
                {
                    await JSRuntime.InvokeVoidAsync("alert", "No se encontró la entrevista asociada al certificado");
                    return;
                }
                
                // Descargar certificado en formato SVG
                var certificateBytes = await InterviewApi.DownloadCertificateAsync(report.InterviewId, "svg");
                
                // Crear blob y descargar
                var fileName = $"certificate_{certificateId}.svg";
                await JSRuntime.InvokeVoidAsync("downloadFile", fileName, "image/svg+xml", certificateBytes);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error downloading certificate: {ex.Message}");
                await JSRuntime.InvokeVoidAsync("alert", $"Error al descargar certificado: {ex.Message}");
            }
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

