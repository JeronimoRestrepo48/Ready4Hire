/**
 * Gamification Redux Slice
 */

import {createSlice, createAsyncThunk} from '@reduxjs/toolkit';
import {apiClient} from '../../services/api/ApiClient';
import {GamificationState, UserStats, Badge, UserBadge, LeaderboardEntry, Game} from '../../types';

const initialState: GamificationState = {
  stats: null,
  badges: [],
  userBadges: [],
  leaderboard: [],
  games: [],
  loading: false,
  error: null,
};

// Fetch User Stats
export const fetchUserStats = createAsyncThunk(
  'gamification/fetchStats',
  async (userId: string) => {
    const response = await apiClient.get<UserStats>(`/gamification/stats/${userId}`);
    return response;
  }
);

// Fetch Badges
export const fetchBadges = createAsyncThunk('gamification/fetchBadges', async () => {
  const response = await apiClient.get<Badge[]>('/badges');
  return response;
});

// Fetch User Badges
export const fetchUserBadges = createAsyncThunk(
  'gamification/fetchUserBadges',
  async (userId: string) => {
    const response = await apiClient.get(`/users/${userId}/badges`);
    return response;
  }
);

// Fetch Leaderboard
export const fetchLeaderboard = createAsyncThunk('gamification/fetchLeaderboard', async () => {
  const response = await apiClient.get<LeaderboardEntry[]>('/gamification/leaderboard');
  return response;
});

// Fetch Games
export const fetchGames = createAsyncThunk('gamification/fetchGames', async () => {
  const response = await apiClient.get<Game[]>('/games');
  return response;
});

// Slice
const gamificationSlice = createSlice({
  name: 'gamification',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchUserStats.fulfilled, (state, action) => {
        state.stats = action.payload;
      })
      .addCase(fetchBadges.fulfilled, (state, action) => {
        state.badges = action.payload;
      })
      .addCase(fetchUserBadges.fulfilled, (state, action) => {
        state.userBadges = action.payload as UserBadge[];
      })
      .addCase(fetchLeaderboard.fulfilled, (state, action) => {
        state.leaderboard = action.payload;
      })
      .addCase(fetchGames.fulfilled, (state, action) => {
        state.games = action.payload;
      });
  },
});

export default gamificationSlice.reducer;

