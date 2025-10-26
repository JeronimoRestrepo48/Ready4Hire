using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using System.Collections.Generic;
using Ready4Hire.MVVM.Models;

namespace Ready4Hire.Services
{
    public class GamificationService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public GamificationService(HttpClient httpClient)
        {
            _httpClient = httpClient;
            _baseUrl = "http://localhost:8001";
        }

        // User Stats
        public async Task<UserStats?> GetUserStatsAsync(string userId)
        {
            return await _httpClient.GetFromJsonAsync<UserStats>(
                $"{_baseUrl}/api/v2/gamification/stats/{userId}");
        }

        // Achievements
        public async Task<List<Achievement>?> GetAchievementsAsync(string userId)
        {
            return await _httpClient.GetFromJsonAsync<List<Achievement>>(
                $"{_baseUrl}/api/v2/gamification/achievements/{userId}");
        }

        // Leaderboard
        public async Task<List<LeaderboardEntry>?> GetLeaderboardAsync(
            string? profession = null, int limit = 100)
        {
            var url = $"{_baseUrl}/api/v2/gamification/leaderboard?limit={limit}";
            if (!string.IsNullOrEmpty(profession))
                url += $"&profession={profession}";
            
            return await _httpClient.GetFromJsonAsync<List<LeaderboardEntry>>(url);
        }

        // Games
        public async Task<List<GameInfo>?> GetAvailableGamesAsync(
            string? profession = null, string? difficulty = null)
        {
            var url = $"{_baseUrl}/api/v2/games";
            var queries = new List<string>();
            
            if (!string.IsNullOrEmpty(profession))
                queries.Add($"profession={profession}");
            if (!string.IsNullOrEmpty(difficulty))
                queries.Add($"difficulty={difficulty}");
                
            if (queries.Count > 0)
                url += "?" + string.Join("&", queries);
            
            return await _httpClient.GetFromJsonAsync<List<GameInfo>>(url);
        }

        // Professions
        public async Task<List<Profession>?> GetProfessionsAsync()
        {
            return await _httpClient.GetFromJsonAsync<List<Profession>>(
                $"{_baseUrl}/api/v2/professions");
        }

        // Skills
        public async Task<Dictionary<string, Dictionary<string, string>>?> GetSkillsAsync()
        {
            return await _httpClient.GetFromJsonAsync<Dictionary<string, Dictionary<string, string>>>(
                $"{_baseUrl}/api/v2/skills");
        }
    }
}

