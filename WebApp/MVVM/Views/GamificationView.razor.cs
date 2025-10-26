using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using Ready4Hire.MVVM.Models;
using Ready4Hire.Services;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Ready4Hire.MVVM.Views
{
    public partial class GamificationView : ComponentBase
    {
        [Inject]
        public GamificationService GamificationService { get; set; } = null!;
        
        [Inject]
        public NavigationManager Navigation { get; set; } = null!;

        private UserStats? stats;
        private List<Achievement> achievements = new();
        private List<GameInfo> games = new();
        private List<LeaderboardEntry> leaderboard = new();
        
        private string userId = "1"; // TODO: Get from auth service
        private string activeTab = "stats";
        private bool isLoading = true;
        private bool chartsInitialized = false;

        protected override async Task OnInitializedAsync()
        {
            await LoadData();
        }

        private async Task LoadData()
        {
            isLoading = true;
            StateHasChanged();

            try
            {
                // Load all data in parallel
                var statsTask = GamificationService.GetUserStatsAsync(userId);
                var achievementsTask = GamificationService.GetAchievementsAsync(userId);
                var gamesTask = GamificationService.GetAvailableGamesAsync();
                var leaderboardTask = GamificationService.GetLeaderboardAsync(limit: 50);

                await Task.WhenAll(statsTask, achievementsTask, gamesTask, leaderboardTask);

                stats = await statsTask;
                achievements = await achievementsTask ?? new();
                games = await gamesTask ?? new();
                leaderboard = await leaderboardTask ?? new();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading gamification data: {ex.Message}");
            }
            finally
            {
                isLoading = false;
                StateHasChanged();
            }
        }

        private double GetXpPercentage()
        {
            if (stats == null) return 0;
            int nextLevelXp = GetNextLevelXp();
            int currentLevelXp = stats.Level * 1000;
            int xpInCurrentLevel = stats.Experience - currentLevelXp;
            return (double)xpInCurrentLevel / 1000 * 100;
        }

        private int GetNextLevelXp()
        {
            if (stats == null) return 1000;
            return (stats.Level + 1) * 1000;
        }

        private string GetGameIcon(string gameType)
        {
            return gameType.ToLower() switch
            {
                "code_challenge" => "💻",
                "quick_quiz" => "⚡",
                "scenario_simulator" => "🎯",
                "speed_round" => "⏱️",
                "skill_builder" => "🛠️",
                "problem_solver" => "🧩",
                "debugging_challenge" => "🐛",
                "system_design" => "🏗️",
                "algorithm_race" => "🏁",
                "code_review" => "👀",
                "api_builder" => "🔌",
                "database_quest" => "🗄️",
                "security_audit" => "🔐",
                "performance_optimizer" => "⚡",
                _ => "🎮"
            };
        }

        private string GetRankMedal(int rank)
        {
            return rank switch
            {
                1 => "🥇",
                2 => "🥈",
                3 => "🥉",
                _ => $"#{rank}"
            };
        }

        private async Task StartGame(string gameId, string gameType)
        {
            // Abrir juego en nueva pestaña
            Console.WriteLine($"Starting game: {gameId} ({gameType})");
            
            // Redirigir según el tipo de juego
            var route = gameType.ToLower() switch
            {
                "quick_quiz" => "/game/quick-quiz",
                "code_challenge" => "/chat/0?game=code",
                "scenario_simulator" => "/chat/0?game=scenario",
                "speed_round" => "/game/quick-quiz",
                "skill_builder" => "/chat/0?game=skill",
                "problem_solver" => "/chat/0?game=problem",
                "debugging_challenge" => "/chat/0?game=debug",
                "system_design" => "/chat/0?game=design",
                "algorithm_race" => "/game/quick-quiz?type=algorithm",
                "code_review" => "/chat/0?game=review",
                "api_builder" => "/chat/0?game=api",
                "database_quest" => "/chat/0?game=database",
                "security_audit" => "/chat/0?game=security",
                "performance_optimizer" => "/chat/0?game=performance",
                _ => "/chat/0"
            };
            
            // Abrir en nueva pestaña
            await JSRuntime.InvokeVoidAsync("open", route, "_blank");
        }

        private async Task SwitchTab(string tab)
        {
            activeTab = tab;
            StateHasChanged();
            
            if (tab == "stats" && !chartsInitialized)
            {
                await Task.Delay(100); // Esperar a que se renderice el canvas
                await InitializeCharts();
            }
        }

        private async Task InitializeCharts()
        {
            try
            {
                // Datos de ejemplo - en producción vendrían del backend
                await JSRuntime.InvokeVoidAsync("initializeGamificationCharts");
                chartsInitialized = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing charts: {ex.Message}");
            }
        }
    }
}

