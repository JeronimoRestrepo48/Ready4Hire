using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Ready4Hire.Services;
using Ready4Hire.MVVM.Models;
using System.Text.Json;
using System.Text;

namespace Ready4Hire.MVVM.Views
{
    public partial class SettingsPage : ComponentBase
    {
        [Inject]
        public NavigationManager Navigation { get; set; } = null!;

        [Inject]
        public AuthService AuthService { get; set; } = null!;

        [Inject]
        public InterviewApiService InterviewApi { get; set; } = null!;

        [Inject]
        public Ready4Hire.Services.SettingsApiService SettingsApi { get; set; } = null!;

        [Inject]
        public IJSRuntime JSRuntime { get; set; } = null!;

        private string? errorMessage = null;
        private string? successMessage = null;
        private Ready4Hire.MVVM.Models.User? currentUser = null;

        protected override async Task OnInitializedAsync()
        {
            currentUser = await AuthService.GetCurrentUserAsync();
        }

        private async Task HandleChangePassword()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                // Solicitar contraseña actual y nueva mediante prompt (temporal, se puede mejorar con un modal)
                var currentPassword = await JSRuntime.InvokeAsync<string>("prompt", "Ingresa tu contraseña actual:");
                if (string.IsNullOrEmpty(currentPassword))
                {
                    return;
                }

                var newPassword = await JSRuntime.InvokeAsync<string>("prompt", "Ingresa tu nueva contraseña (mínimo 8 caracteres):");
                if (string.IsNullOrEmpty(newPassword) || newPassword.Length < 8)
                {
                    errorMessage = "La nueva contraseña debe tener al menos 8 caracteres";
                    StateHasChanged();
                    return;
                }

                var confirmPassword = await JSRuntime.InvokeAsync<string>("prompt", "Confirma tu nueva contraseña:");
                if (newPassword != confirmPassword)
                {
                    errorMessage = "Las contraseñas no coinciden";
                    StateHasChanged();
                    return;
                }

                var result = await SettingsApi.ChangePasswordAsync(currentUser.Email, currentPassword, newPassword);
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Contraseña actualizada exitosamente";
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al cambiar la contraseña";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al cambiar contraseña: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandleConfigureNotifications()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                // Obtener configuración actual
                var currentSettings = await SettingsApi.GetNotificationsAsync(currentUser.Id);
                
                var emailNotifications = currentSettings.TryGetProperty("emailNotifications", out var email) ? email.GetBoolean() : true;
                var achievementNotifications = currentSettings.TryGetProperty("achievementNotifications", out var achievement) ? achievement.GetBoolean() : true;
                var interviewReminders = currentSettings.TryGetProperty("interviewReminders", out var reminders) ? reminders.GetBoolean() : true;

                // Mostrar configuración actual y permitir cambios (temporal, se puede mejorar con un modal)
                var newEmailNotifications = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Activar notificaciones por email? (Actual: {(emailNotifications ? "Sí" : "No")})");
                var newAchievementNotifications = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Activar notificaciones de logros? (Actual: {(achievementNotifications ? "Sí" : "No")})");
                var newInterviewReminders = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Activar recordatorios de entrevistas? (Actual: {(interviewReminders ? "Sí" : "No")})");

                var result = await SettingsApi.UpdateNotificationsAsync(
                    currentUser.Id, 
                    newEmailNotifications, 
                    newAchievementNotifications, 
                    newInterviewReminders
                );
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Preferencias de notificaciones actualizadas";
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al actualizar notificaciones";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al configurar notificaciones: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandlePrivacyAndSecurity()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                // Obtener configuración actual
                var currentSettings = await SettingsApi.GetPrivacyAsync(currentUser.Id);
                
                var showProfilePublic = currentSettings.TryGetProperty("showProfilePublic", out var profile) ? profile.GetBoolean() : false;
                var showStatsPublic = currentSettings.TryGetProperty("showStatsPublic", out var stats) ? stats.GetBoolean() : false;
                var allowDataSharing = currentSettings.TryGetProperty("allowDataSharing", out var sharing) ? sharing.GetBoolean() : false;

                // Mostrar configuración actual y permitir cambios (temporal, se puede mejorar con un modal)
                var newShowProfilePublic = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Mostrar perfil público? (Actual: {(showProfilePublic ? "Sí" : "No")})");
                var newShowStatsPublic = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Mostrar estadísticas públicas? (Actual: {(showStatsPublic ? "Sí" : "No")})");
                var newAllowDataSharing = await JSRuntime.InvokeAsync<bool>("confirm", $"¿Permitir compartir datos para mejoras? (Actual: {(allowDataSharing ? "Sí" : "No")})");

                var result = await SettingsApi.UpdatePrivacyAsync(
                    currentUser.Id, 
                    newShowProfilePublic, 
                    newShowStatsPublic, 
                    newAllowDataSharing
                );
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Preferencias de privacidad actualizadas";
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al actualizar privacidad";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al acceder a configuración de privacidad: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandleLanguageAndRegion()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                // Obtener configuración actual
                var currentSettings = await SettingsApi.GetLanguageAsync(currentUser.Id);
                
                var language = currentSettings.TryGetProperty("language", out var lang) ? lang.GetString() ?? "es" : "es";
                var region = currentSettings.TryGetProperty("region", out var reg) ? reg.GetString() ?? "ES" : "ES";
                var timeZone = currentSettings.TryGetProperty("timeZone", out var tz) ? tz.GetString() ?? "Europe/Madrid" : "Europe/Madrid";

                // Solicitar nuevos valores (temporal, se puede mejorar con un modal)
                var newLanguage = await JSRuntime.InvokeAsync<string>("prompt", $"Idioma (es/en): (Actual: {language})");
                if (string.IsNullOrEmpty(newLanguage)) newLanguage = language;
                
                var newRegion = await JSRuntime.InvokeAsync<string>("prompt", $"Región (ES/US/etc): (Actual: {region})");
                if (string.IsNullOrEmpty(newRegion)) newRegion = region;

                var result = await SettingsApi.UpdateLanguageAsync(
                    currentUser.Id, 
                    newLanguage, 
                    newRegion, 
                    timeZone
                );
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Preferencias de idioma actualizadas";
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al actualizar idioma";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al configurar idioma: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandleExportData()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                var email = await AuthService.GetCurrentUserEmailAsync();
                if (string.IsNullOrEmpty(email))
                {
                    errorMessage = "No se pudo obtener el email del usuario";
                    StateHasChanged();
                    return;
                }

                // Convertir email a user_id
                var userId = $"user_{email.Replace("@", "_at_").Replace(".", "_")}";
                
                // Obtener todas las entrevistas completadas
                var interviewsData = await InterviewApi.GetAllUserInterviewsAsync(userId);
                
                // Crear objeto de exportación
                var exportData = new
                {
                    user_email = email,
                    export_date = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                    interviews = interviewsData
                };

                // Convertir a JSON
                var json = JsonSerializer.Serialize(exportData, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });

                // Descargar como archivo JSON
                var fileName = $"ready4hire_export_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json";
                var bytes = Encoding.UTF8.GetBytes(json);
                
                await JSRuntime.InvokeVoidAsync("downloadFile", fileName, "application/json", Convert.ToBase64String(bytes));
                
                successMessage = "✅ Datos exportados correctamente";
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al exportar datos: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandleDownloadCertificates()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                // Navegar a la página de certificados
                Navigation.NavigateTo("/certificates");
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al descargar certificados: {ex.Message}";
            }
        }

        private async Task HandleGenerateProgressReport()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                var email = await AuthService.GetCurrentUserEmailAsync();
                if (string.IsNullOrEmpty(email))
                {
                    errorMessage = "No se pudo obtener el email del usuario";
                    StateHasChanged();
                    return;
                }

                // Convertir email a user_id
                var userId = $"user_{email.Replace("@", "_at_").Replace(".", "_")}";
                
                // Obtener todas las entrevistas completadas
                var interviewsData = await InterviewApi.GetAllUserInterviewsAsync(userId);
                
                // Generar reporte de progreso
                var progressReport = GenerateProgressReportFromInterviews(interviewsData, email);
                
                // Convertir a JSON
                var json = JsonSerializer.Serialize(progressReport, new JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });

                // Descargar como archivo JSON
                var fileName = $"ready4hire_progress_report_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json";
                var bytes = Encoding.UTF8.GetBytes(json);
                
                await JSRuntime.InvokeVoidAsync("downloadFile", fileName, "application/json", Convert.ToBase64String(bytes));
                
                successMessage = "✅ Reporte de progreso generado correctamente";
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al generar reporte: {ex.Message}";
                StateHasChanged();
            }
        }

        private object GenerateProgressReportFromInterviews(JsonElement interviewsData, string email)
        {
            var interviews = new List<object>();
            var totalInterviews = 0;
            var totalScore = 0.0;
            var rolesCount = new Dictionary<string, int>();
            var certificatesCount = 0;

            if (interviewsData.TryGetProperty("interviews", out var interviewsElement))
            {
                foreach (var interview in interviewsElement.EnumerateArray())
                {
                    totalInterviews++;
                    
                    if (interview.TryGetProperty("average_score", out var scoreElement))
                    {
                        totalScore += scoreElement.GetDouble();
                    }
                    
                    if (interview.TryGetProperty("role", out var roleElement))
                    {
                        var role = roleElement.GetString() ?? "Unknown";
                        rolesCount[role] = rolesCount.GetValueOrDefault(role, 0) + 1;
                    }
                    
                    if (interview.TryGetProperty("has_certificate", out var certElement) && certElement.GetBoolean())
                    {
                        certificatesCount++;
                    }
                    
                    interviews.Add(new
                    {
                        interview_id = interview.TryGetProperty("interview_id", out var id) ? id.GetString() : "",
                        role = interview.TryGetProperty("role", out var r) ? r.GetString() : "",
                        mode = interview.TryGetProperty("mode", out var m) ? m.GetString() : "",
                        average_score = interview.TryGetProperty("average_score", out var s) ? s.GetDouble() : 0.0,
                        completed_at = interview.TryGetProperty("completed_at", out var c) ? c.GetString() : ""
                    });
                }
            }

            var averageScore = totalInterviews > 0 ? totalScore / totalInterviews : 0.0;

            return new
            {
                user_email = email,
                report_date = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                summary = new
                {
                    total_interviews = totalInterviews,
                    average_score = Math.Round(averageScore, 2),
                    certificates_earned = certificatesCount,
                    roles_practiced = rolesCount.Keys.ToList(),
                    interviews_by_role = rolesCount
                },
                interviews = interviews
            };
        }

        private async Task HandleDeleteAllData()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                var confirmed = await JSRuntime.InvokeAsync<bool>("confirm", "⚠️ ¿Estás seguro de que deseas eliminar todos tus datos? Esta acción no se puede deshacer.\n\nSe eliminarán:\n- Todas tus entrevistas\n- Todos tus chats y mensajes\n- Todos tus badges\n- Todas tus estadísticas\n\nTu cuenta permanecerá activa.");
                
                if (!confirmed)
                {
                    return;
                }

                var result = await SettingsApi.DeleteAllDataAsync(currentUser.Id);
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Todos los datos han sido eliminados";
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al eliminar datos";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al eliminar datos: {ex.Message}";
                StateHasChanged();
            }
        }

        private async Task HandleDeleteAccount()
        {
            try
            {
                errorMessage = null;
                successMessage = null;
                
                if (currentUser == null)
                {
                    errorMessage = "No se pudo obtener la información del usuario";
                    StateHasChanged();
                    return;
                }

                var confirmed = await JSRuntime.InvokeAsync<bool>("confirm", "⚠️⚠️⚠️ PELIGRO ⚠️⚠️⚠️\n\n¿Estás COMPLETAMENTE seguro de que deseas eliminar tu cuenta permanentemente?\n\nEsta acción:\n- Eliminará TODOS tus datos\n- Eliminará tu cuenta\n- NO SE PUEDE DESHACER\n\nEscribe 'ELIMINAR' para confirmar:");
                
                if (!confirmed)
                {
                    return;
                }

                var confirmationText = await JSRuntime.InvokeAsync<string>("prompt", "Escribe 'ELIMINAR' para confirmar la eliminación permanente:");
                if (confirmationText != "ELIMINAR")
                {
                    errorMessage = "Confirmación incorrecta. La eliminación fue cancelada.";
                    StateHasChanged();
                    return;
                }

                var result = await SettingsApi.DeleteAccountAsync(currentUser.Id);
                
                if (result.TryGetProperty("success", out var success) && success.GetBoolean())
                {
                    successMessage = "✅ Cuenta eliminada permanentemente";
                    // Redirigir al login después de un breve delay
                    await Task.Delay(2000);
                    await AuthService.LogoutAsync();
                    Navigation.NavigateTo("/", true);
                }
                else
                {
                    errorMessage = result.TryGetProperty("message", out var msg) ? msg.GetString() : "Error al eliminar cuenta";
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                errorMessage = $"Error al eliminar cuenta: {ex.Message}";
                StateHasChanged();
            }
        }
    }
}

