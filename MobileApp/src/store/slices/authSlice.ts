/**
 * Auth Redux Slice
 * Handles authentication state and actions
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';
import {apiClient} from '../../services/api/ApiClient';
import {webAppClient} from '../../services/api/WebAppClient';
import {AuthState, User, LoginCredentials, RegisterData} from '../../types';
import AsyncStorage from '@react-native-async-storage/async-storage';

const initialState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: null,
  loading: false,
  error: null,
};

// Async Thunks
export const loginUser = createAsyncThunk(
  'auth/login',
  async (credentials: LoginCredentials, {rejectWithValue}) => {
    try {
      // Step 1: WebApp login → returns user info
      const webAppResult = await webAppClient.post<{ success: boolean; message: string; user: any }>(
        '/Auth/login',
        credentials
      );

      const user: User = {
        id: String(webAppResult.user.id),
        email: webAppResult.user.email,
        name: webAppResult.user.name,
        lastName: webAppResult.user.lastName,
        country: webAppResult.user.country,
        profession: webAppResult.user.job,
        skills: webAppResult.user.skills,
        softSkills: webAppResult.user.softskills,
        interests: webAppResult.user.interests,
      };

      // Step 2: Get JWT tokens from FastAPI backend
      // Use email as username for FastAPI authentication
      let accessToken: string | null = null;
      let refreshToken: string | null = null;

      try {
        const jwtResponse = await apiClient.post<{
          access_token: string;
          refresh_token: string;
          token_type: string;
          expires_in: number;
        }>('/auth/login', {
          username: credentials.email, // FastAPI uses 'username' field
          password: credentials.password,
        });

        accessToken = jwtResponse.access_token;
        refreshToken = jwtResponse.refresh_token || null;

        // Store tokens in ApiClient (which uses Keychain)
        if (accessToken && refreshToken) {
          apiClient.setTokens(accessToken, refreshToken);
        } else if (accessToken) {
          apiClient.setToken(accessToken);
        }
      } catch (jwtError: any) {
        // If FastAPI login fails, log but don't fail the entire login
        // User can still use WebApp features, but FastAPI features won't work
        console.warn('Failed to get JWT from FastAPI backend:', jwtError.message);
        // Continue with user data, tokens will be null
      }

      // Persist user data to AsyncStorage
      await AsyncStorage.setItem('@Ready4Hire:user_data', JSON.stringify(user));

      return {
        user,
        token: accessToken || (null as unknown as string),
      } as { user: User; token: string };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Login failed');
    }
  }
);

export const registerUser = createAsyncThunk(
  'auth/register',
  async (data: RegisterData, {rejectWithValue}) => {
    try {
      // Step 1: Register in WebApp
      const webAppResult = await webAppClient.post<{
        success: boolean;
        message: string;
        userId: number;
        email: string;
        name: string;
      }>('/Auth/register', {
        Email: data.email,
        Password: data.password,
        Name: data.name,
        LastName: data.lastName,
        Country: data.country || 'N/A',
        Job: data.job || 'Candidate',
        Skills: data.skills || [],
        Softskills: data.softskills || [],
        Interests: data.interests || [],
      });

      const user: User = {
        id: String(webAppResult.userId),
        email: webAppResult.email,
        name: webAppResult.name,
        lastName: data.lastName,
        country: data.country,
        profession: data.job,
        skills: data.skills,
        softSkills: data.softskills,
        interests: data.interests,
      };

      // Step 2: Get JWT tokens from FastAPI backend (using same credentials)
      let accessToken: string | null = null;
      let refreshToken: string | null = null;

      try {
        const jwtResponse = await apiClient.post<{
          access_token: string;
          refresh_token: string;
          token_type: string;
          expires_in: number;
        }>('/auth/login', {
          username: data.email, // FastAPI uses 'username' field
          password: data.password,
        });

        accessToken = jwtResponse.access_token;
        refreshToken = jwtResponse.refresh_token || null;

        // Store tokens in ApiClient (which uses Keychain)
        if (accessToken && refreshToken) {
          apiClient.setTokens(accessToken, refreshToken);
        } else if (accessToken) {
          apiClient.setToken(accessToken);
        }
      } catch (jwtError: any) {
        // If FastAPI login fails after registration, log but don't fail
        console.warn('Failed to get JWT from FastAPI backend after registration:', jwtError.message);
        // Continue with user data, tokens will be null
      }

      await AsyncStorage.setItem('@Ready4Hire:user_data', JSON.stringify(user));

      return {
        user,
        token: accessToken || (null as unknown as string),
      } as { user: User; token: string };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Registration failed');
    }
  }
);

export const loadUserFromStorage = createAsyncThunk(
  'auth/loadUser',
  async () => {
    try {
      const userData = await AsyncStorage.getItem('@Ready4Hire:user_data');
      
      // Load tokens from Keychain (via ApiClient)
      await apiClient.loadToken();
      
      // Try to get token from ApiClient (which loads from Keychain)
      // Since ApiClient stores tokens internally, we need to check if they exist
      // For now, we'll check if user exists and assume tokens are loaded in ApiClient
      if (userData) {
        const user = JSON.parse(userData);
        // Tokens are now stored in Keychain and loaded in ApiClient
        // We don't need to pass token to state since ApiClient handles it
        return {
          user,
          token: null as unknown as string, // Token is handled by ApiClient internally
        };
      }
      return null;
    } catch (error: any) {
      console.error('Error loading user from storage:', error);
      return null;
    }
  }
);

export const logoutUser = createAsyncThunk(
  'auth/logout',
  async () => {
    // Clear tokens from Keychain and AsyncStorage
    await apiClient.clearToken();
    await AsyncStorage.removeItem('@Ready4Hire:user_data');
  }
);

// Slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setUser: (state, action: PayloadAction<User>) => {
      state.user = action.payload;
    },
  },
  extraReducers: (builder) => {
    // Login
    builder
      .addCase(loginUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.token = action.payload.token;
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Register
    builder
      .addCase(registerUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(registerUser.fulfilled, (state, action) => {
        state.loading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.token = action.payload.token;
      })
      .addCase(registerUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Load User from Storage
    builder
      .addCase(loadUserFromStorage.pending, (state) => {
        state.loading = true;
      })
      .addCase(loadUserFromStorage.fulfilled, (state, action) => {
        state.loading = false;
        if (action.payload) {
          state.isAuthenticated = true;
          state.user = action.payload.user;
          state.token = action.payload.token;
        }
      })
      .addCase(loadUserFromStorage.rejected, (state) => {
        state.loading = false;
      });

    // Logout
    builder
      .addCase(logoutUser.fulfilled, (state) => {
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
        state.error = null;
      });
  },
});

export const {clearError, setUser} = authSlice.actions;
export default authSlice.reducer;

