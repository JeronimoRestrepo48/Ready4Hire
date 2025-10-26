using Microsoft.AspNetCore.Components;
using Microsoft.EntityFrameworkCore;
using Ready4Hire.Data;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Ready4Hire.MVVM.Views
{
    public partial class ProfileView : ComponentBase
    {
        [Inject]
        public IDbContextFactory<AppDbContext> DbFactory { get; set; } = null!;

        [Inject]
        public AuthService AuthService { get; set; } = null!;

        [Inject]
        public NavigationManager Navigation { get; set; } = null!;

        private User? user;
        private List<UserBadge> userBadges = new();
        private List<Badge> availableBadges = new();
        
        private string activeTab = "info";
        private bool isLoading = true;

        // Filtros de badges
        private string badgeFilterRarity = "";
        private string badgeFilterCategory = "";
        private bool showOnlyUnlocked = false;

        // Modal de agregar habilidades
        private bool showAddSkillModal = false;
        private string currentSkillType = "";
        private string newSkillInput = "";
        private List<string> filteredSuggestions = new();
        
        // Sugerencias de habilidades
        private readonly List<string> technicalSkillsSuggestions = new()
        {
            "Python", "C#", ".NET", "JavaScript", "TypeScript", "Java", "React", "Angular", "Vue.js",
            "Node.js", "ASP.NET", "Blazor", "SQL", "PostgreSQL", "MongoDB", "Docker", "Kubernetes",
            "AWS", "Azure", "GCP", "Git", "CI/CD", "REST API", "GraphQL", "Microservicios", "TDD",
            "Agile", "Scrum", "DevOps", "Machine Learning", "Data Science", "Cybersecurity",
            "Cloud Computing", "Big Data", "Blockchain", "IoT", "Mobile Development", "UI/UX Design"
        };
        
        private readonly List<string> softSkillsSuggestions = new()
        {
            "Liderazgo", "Trabajo en equipo", "Comunicación", "Resolución de problemas",
            "Pensamiento crítico", "Creatividad", "Adaptabilidad", "Gestión del tiempo",
            "Inteligencia emocional", "Negociación", "Presentaciones", "Mentoring",
            "Delegación", "Toma de decisiones", "Empatía", "Colaboración", "Iniciativa",
            "Gestión de conflictos", "Organización", "Proactividad"
        };
        
        private readonly List<string> interestsSuggestions = new()
        {
            "Inteligencia Artificial", "Machine Learning", "Desarrollo Web", "Desarrollo Móvil",
            "Cloud Computing", "Cyberseguridad", "Blockchain", "Data Science", "DevOps",
            "Arquitectura de Software", "IoT", "Realidad Virtual", "Realidad Aumentada",
            "Automatización", "Big Data", "Análisis de Datos", "Product Management",
            "UX Research", "Design Thinking", "Innovación", "Startups", "Open Source"
        };

        protected override async Task OnInitializedAsync()
        {
            await LoadUserData();
        }

        private async Task LoadUserData()
        {
            isLoading = true;
            StateHasChanged();

            try
            {
                var currentUser = await AuthService.GetCurrentUserAsync();
                if (currentUser == null)
                {
                    Navigation.NavigateTo("/", true);
                    return;
                }

                using var db = await DbFactory.CreateDbContextAsync();

                // Cargar usuario con badges
                user = await db.Users
                    .Include(u => u.Badges)
                    .ThenInclude(ub => ub.Badge)
                    .FirstOrDefaultAsync(u => u.Id == currentUser.Id);

                if (user != null)
                {
                    userBadges = user.Badges;

                    // Cargar todos los badges disponibles
                    availableBadges = await db.Badges.ToListAsync();

                    // Si el usuario no tiene badges, crearlos
                    if (userBadges.Count == 0 && availableBadges.Count > 0)
                    {
                        foreach (var badge in availableBadges)
                        {
                            var userBadge = new UserBadge
                            {
                                UserId = user.Id,
                                BadgeId = badge.Id,
                                IsUnlocked = false,
                                Progress = 0
                            };
                            db.UserBadges.Add(userBadge);
                        }
                        await db.SaveChangesAsync();

                        // Recargar badges
                        user = await db.Users
                            .Include(u => u.Badges)
                            .ThenInclude(ub => ub.Badge)
                            .FirstOrDefaultAsync(u => u.Id == currentUser.Id);

                        if (user != null)
                        {
                            userBadges = user.Badges;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading user data: {ex.Message}");
            }
            finally
            {
                isLoading = false;
                StateHasChanged();
            }
        }

        private string GetUserInitials()
        {
            if (user == null) return "U";

            var initials = "";
            if (!string.IsNullOrEmpty(user.Name))
                initials += user.Name[0];
            if (!string.IsNullOrEmpty(user.LastName))
                initials += user.LastName[0];

            return string.IsNullOrEmpty(initials) ? "U" : initials.ToUpper();
        }

        private int GetNextLevelXp()
        {
            if (user == null) return 1000;
            return (user.Level + 1) * 1000;
        }

        private double GetXpProgress()
        {
            if (user == null) return 0;
            int currentLevelXp = user.Level * 1000;
            int xpInCurrentLevel = user.Experience - currentLevelXp;
            return (double)xpInCurrentLevel / 1000 * 100;
        }

        private int GetWinRate()
        {
            if (user == null || user.TotalGamesPlayed == 0) return 0;
            return (int)((double)user.TotalGamesWon / user.TotalGamesPlayed * 100);
        }

        private IEnumerable<UserBadge> GetFilteredBadges()
        {
            var filtered = userBadges.AsEnumerable();

            if (showOnlyUnlocked)
            {
                filtered = filtered.Where(ub => ub.IsUnlocked);
            }

            if (!string.IsNullOrEmpty(badgeFilterRarity))
            {
                filtered = filtered.Where(ub =>
                {
                    var badge = availableBadges.FirstOrDefault(b => b.Id == ub.BadgeId);
                    return badge != null && badge.Rarity == badgeFilterRarity;
                });
            }

            if (!string.IsNullOrEmpty(badgeFilterCategory))
            {
                filtered = filtered.Where(ub =>
                {
                    var badge = availableBadges.FirstOrDefault(b => b.Id == ub.BadgeId);
                    return badge != null && badge.Category == badgeFilterCategory;
                });
            }

            // Ordenar: desbloqueados primero, luego por progreso
            return filtered.OrderByDescending(ub => ub.IsUnlocked)
                          .ThenByDescending(ub => ub.Progress);
        }

        private string GetRarityName(string rarity)
        {
            return rarity switch
            {
                "common" => "COMÚN",
                "rare" => "RARO",
                "epic" => "ÉPICO",
                "legendary" => "LEGENDARIO",
                _ => rarity.ToUpper()
            };
        }

        private string GetRequirementText(Badge badge)
        {
            if (badge.RequirementType == null) return "";

            var requirement = badge.RequirementType switch
            {
                "interviews_completed" => $"Completa {badge.RequirementValue} entrevistas",
                "games_played" => $"Juega {badge.RequirementValue} juegos",
                "games_won" => $"Gana {badge.RequirementValue} juegos",
                "streak_days" => $"Mantén {badge.RequirementValue} días de racha",
                "level_reached" => $"Alcanza el nivel {badge.RequirementValue}",
                "total_points" => $"Acumula {badge.RequirementValue:N0} puntos",
                "code_challenges_won" => $"Gana {badge.RequirementValue} desafíos de código",
                "technical_correct_answers" => $"Responde {badge.RequirementValue} preguntas técnicas correctamente",
                "soft_skills_interviews" => $"Completa {badge.RequirementValue} entrevistas de soft skills",
                "scenarios_completed" => $"Completa {badge.RequirementValue} escenarios",
                "perfect_interviews" => $"Consigue {badge.RequirementValue} entrevistas perfectas (100%)",
                _ => $"{badge.RequirementType}: {badge.RequirementValue}"
            };

            return requirement;
        }

        private void ShowAddSkillModal(string skillType)
        {
            currentSkillType = skillType;
            newSkillInput = "";
            filteredSuggestions.Clear();
            showAddSkillModal = true;
        }

        private void HideAddSkillModal()
        {
            showAddSkillModal = false;
            newSkillInput = "";
            filteredSuggestions.Clear();
        }

        private void FilterSkillSuggestions()
        {
            if (string.IsNullOrWhiteSpace(newSkillInput))
            {
                filteredSuggestions.Clear();
                return;
            }

            var suggestions = currentSkillType switch
            {
                "technical" => technicalSkillsSuggestions,
                "soft" => softSkillsSuggestions,
                "interest" => interestsSuggestions,
                _ => new List<string>()
            };

            filteredSuggestions = suggestions
                .Where(s => s.Contains(newSkillInput, StringComparison.OrdinalIgnoreCase))
                .OrderBy(s => s)
                .ToList();

            StateHasChanged();
        }

        private void SelectSuggestion(string suggestion)
        {
            newSkillInput = suggestion;
            filteredSuggestions.Clear();
            StateHasChanged();
        }

        private async Task AddNewSkill()
        {
            if (string.IsNullOrWhiteSpace(newSkillInput) || user == null)
                return;

            try
            {
                using var db = await DbFactory.CreateDbContextAsync();
                var dbUser = await db.Users.FirstOrDefaultAsync(u => u.Id == user.Id);
                
                if (dbUser != null)
                {
                    var skillToAdd = newSkillInput.Trim();
                    
                    switch (currentSkillType)
                    {
                        case "technical":
                            if (!dbUser.Skills.Contains(skillToAdd))
                            {
                                dbUser.Skills.Add(skillToAdd);
                                user.Skills.Add(skillToAdd);
                            }
                            break;
                        case "soft":
                            if (!dbUser.Softskills.Contains(skillToAdd))
                            {
                                dbUser.Softskills.Add(skillToAdd);
                                user.Softskills.Add(skillToAdd);
                            }
                            break;
                        case "interest":
                            if (!dbUser.Interests.Contains(skillToAdd))
                            {
                                dbUser.Interests.Add(skillToAdd);
                                user.Interests.Add(skillToAdd);
                            }
                            break;
                    }
                    
                    await db.SaveChangesAsync();
                }
                
                HideAddSkillModal();
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error adding skill: {ex.Message}");
            }
        }

        private async Task RemoveSkill(string skillType, string skill)
        {
            if (user == null)
                return;

            try
            {
                using var db = await DbFactory.CreateDbContextAsync();
                var dbUser = await db.Users.FirstOrDefaultAsync(u => u.Id == user.Id);
                
                if (dbUser != null)
                {
                    switch (skillType)
                    {
                        case "technical":
                            dbUser.Skills.Remove(skill);
                            user.Skills.Remove(skill);
                            break;
                        case "soft":
                            dbUser.Softskills.Remove(skill);
                            user.Softskills.Remove(skill);
                            break;
                        case "interest":
                            dbUser.Interests.Remove(skill);
                            user.Interests.Remove(skill);
                            break;
                    }
                    
                    await db.SaveChangesAsync();
                }
                
                StateHasChanged();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error removing skill: {ex.Message}");
            }
        }

        private string GetSkillTypeLabel(string skillType)
        {
            return skillType switch
            {
                "technical" => "Habilidad Técnica",
                "soft" => "Habilidad Blanda",
                "interest" => "Interés",
                _ => "Habilidad"
            };
        }
    }
}

