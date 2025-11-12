/**
 * API Client for Ready4Hire Mobile App
 * Handles all HTTP requests to the backend
 */

import axios, {AxiosInstance, AxiosRequestConfig, AxiosError} from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Keychain from 'react-native-keychain';

// Extend AxiosRequestConfig to include metadata and retry flag
declare module 'axios' {
  export interface AxiosRequestConfig {
    metadata?: {
      startTime?: number;
    };
    _retry?: boolean;
  }
}

// Configuration
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8001';
const API_VERSION = process.env.API_VERSION || 'v2';
const API_TIMEOUT = 30000; // 30 seconds

class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;
  private refreshToken: string | null = null;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (value?: any) => void;
    reject: (reason?: any) => void;
  }> = [];

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api/${API_VERSION}`,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      async (config) => {
        // Add auth token if available
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }

        // Add request timestamp
        config.metadata = {startTime: Date.now()};

        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        const duration = response.config.metadata?.startTime
          ? Date.now() - response.config.metadata.startTime
          : 0;
        console.log(`✅ ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`);
        return response;
      },
      async (error: AxiosError) => {
        const duration = error.config?.metadata?.startTime
          ? Date.now() - error.config.metadata.startTime
          : 0;

        console.error(`❌ ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`);

        const originalRequest = error.config;

        // Handle 401 Unauthorized - try to refresh token
        if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
          if (this.isRefreshing) {
            // If already refreshing, queue this request
            return new Promise((resolve, reject) => {
              this.failedQueue.push({resolve, reject});
            })
              .then((token) => {
                originalRequest.headers.Authorization = `Bearer ${token}`;
                return this.client(originalRequest);
              })
              .catch((err) => {
                return Promise.reject(err);
              });
          }

          originalRequest._retry = true;
          this.isRefreshing = true;

          try {
            const newTokens = await this.refreshAccessToken();
            if (newTokens) {
              // Update tokens
              this.token = newTokens.access_token;
              this.refreshToken = newTokens.refresh_token;
              await this.saveTokens(newTokens.access_token, newTokens.refresh_token);

              // Update original request header
              originalRequest.headers.Authorization = `Bearer ${newTokens.access_token}`;

              // Process queued requests
              this.processQueue(null, newTokens.access_token);

              // Retry original request
              return this.client(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed - clear tokens and process queue with error
            this.processQueue(refreshError, null);
            this.clearTokens();
            return Promise.reject(refreshError);
          } finally {
            this.isRefreshing = false;
          }
        }

        // Handle other errors
        if (error.response) {
          // Server responded with error
          const status = error.response.status;
          const data = error.response.data;

          switch (status) {
            case 403:
              // Forbidden
              break;
            case 404:
              // Not found
              break;
            case 429:
              // Rate limit exceeded
              break;
            case 500:
              // Internal server error
              break;
          }

          return Promise.reject({
            error: 'API_ERROR',
            message: (data as any)?.message || error.message,
            status,
            timestamp: new Date().toISOString(),
          });
        } else if (error.request) {
          // Request made but no response
          return Promise.reject({
            error: 'NETWORK_ERROR',
            message: 'No response from server. Please check your connection.',
            timestamp: new Date().toISOString(),
          });
        }

        return Promise.reject(error);
      }
    );
  }

  /**
   * Set authentication tokens (access + refresh)
   */
  setTokens(accessToken: string, refreshToken: string): void {
    this.token = accessToken;
    this.refreshToken = refreshToken;
    this.saveTokens(accessToken, refreshToken);
  }

  /**
   * Set authentication token (legacy - for backwards compatibility)
   */
  setToken(token: string): void {
    this.token = token;
    this.saveToken(token);
  }

  /**
   * Clear all authentication tokens
   */
  clearToken(): Promise<void> {
    return this.clearTokens();
  }

  /**
   * Clear all tokens (internal)
   */
  private async clearTokens(): Promise<void> {
    this.token = null;
    this.refreshToken = null;
    try {
      // Clear from Keychain (secure storage)
      await Keychain.resetGenericPassword();
      // Also clear from AsyncStorage (fallback)
      await AsyncStorage.multiRemove([
        '@Ready4Hire:auth_token',
        '@Ready4Hire:refresh_token',
      ]);
    } catch (error) {
      console.error('Failed to clear tokens:', error);
    }
  }

  /**
   * Save tokens to Keychain (secure storage)
   */
  private async saveTokens(accessToken: string, refreshToken: string): Promise<void> {
    try {
      // Store tokens in Keychain (encrypted)
      const tokensJSON = JSON.stringify({
        access_token: accessToken,
        refresh_token: refreshToken,
      });
      
      await Keychain.setGenericPassword('Ready4Hire_Tokens', tokensJSON, {
        service: 'com.ready4hire.tokens',
        accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED,
      });
      
      // Also save to AsyncStorage as fallback (for migration period)
      await AsyncStorage.multiSet([
        ['@Ready4Hire:auth_token', accessToken],
        ['@Ready4Hire:refresh_token', refreshToken],
      ]);
    } catch (error) {
      console.error('Failed to save tokens to Keychain:', error);
      // Fallback to AsyncStorage if Keychain fails
      try {
        await AsyncStorage.multiSet([
          ['@Ready4Hire:auth_token', accessToken],
          ['@Ready4Hire:refresh_token', refreshToken],
        ]);
      } catch (fallbackError) {
        console.error('Failed to save tokens to AsyncStorage:', fallbackError);
      }
    }
  }

  /**
   * Save token to Keychain (legacy - for backwards compatibility)
   */
  private async saveToken(token: string): Promise<void> {
    try {
      // Try to load existing refresh token from Keychain
      const existingTokens = await this.loadTokensFromKeychain();
      const refreshToken = existingTokens?.refresh_token || null;
      
      if (refreshToken) {
        await this.saveTokens(token, refreshToken);
      } else {
        // If no refresh token, just save access token
        await Keychain.setGenericPassword('Ready4Hire_AccessToken', token, {
          service: 'com.ready4hire.tokens',
          accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED,
        });
        await AsyncStorage.setItem('@Ready4Hire:auth_token', token);
      }
    } catch (error) {
      console.error('Failed to save token:', error);
      // Fallback to AsyncStorage
      try {
        await AsyncStorage.setItem('@Ready4Hire:auth_token', token);
      } catch (fallbackError) {
        console.error('Failed to save token to AsyncStorage:', fallbackError);
      }
    }
  }

  /**
   * Load tokens from Keychain (secure storage)
   */
  async loadToken(): Promise<void> {
    try {
      const tokens = await this.loadTokensFromKeychain();
      
      if (tokens) {
        this.token = tokens.access_token;
        this.refreshToken = tokens.refresh_token;
      } else {
        // Fallback: Try to migrate from AsyncStorage
        await this.migrateTokensFromAsyncStorage();
      }
    } catch (error) {
      console.error('Failed to load tokens from Keychain:', error);
      // Fallback to AsyncStorage
      await this.migrateTokensFromAsyncStorage();
    }
  }

  /**
   * Load tokens from Keychain
   */
  private async loadTokensFromKeychain(): Promise<{
    access_token: string;
    refresh_token: string;
  } | null> {
    try {
      const credentials = await Keychain.getGenericPassword({
        service: 'com.ready4hire.tokens',
      });

      if (credentials && credentials.password) {
        const tokens = JSON.parse(credentials.password);
        return {
          access_token: tokens.access_token,
          refresh_token: tokens.refresh_token,
        };
      }
      return null;
    } catch (error) {
      console.error('Failed to load tokens from Keychain:', error);
      return null;
    }
  }

  /**
   * Migrate tokens from AsyncStorage to Keychain (one-time migration)
   */
  private async migrateTokensFromAsyncStorage(): Promise<void> {
    try {
      const [accessToken, refreshToken] = await AsyncStorage.multiGet([
        '@Ready4Hire:auth_token',
        '@Ready4Hire:refresh_token',
      ]);

      if (accessToken[1]) {
        this.token = accessToken[1];
      }
      if (refreshToken[1]) {
        this.refreshToken = refreshToken[1];
      }

      // If we have tokens, migrate them to Keychain
      if (accessToken[1] && refreshToken[1]) {
        await this.saveTokens(accessToken[1], refreshToken[1]);
        console.log('✅ Migrated tokens from AsyncStorage to Keychain');
      }
    } catch (error) {
      console.error('Failed to migrate tokens:', error);
    }
  }

  /**
   * Refresh access token using refresh token
   */
  private async refreshAccessToken(): Promise<{access_token: string; refresh_token: string} | null> {
    if (!this.refreshToken) {
      // Try to load from Keychain first
      const tokens = await this.loadTokensFromKeychain();
      if (tokens?.refresh_token) {
        this.refreshToken = tokens.refresh_token;
      } else {
        // Fallback to AsyncStorage
        const storedRefreshToken = await AsyncStorage.getItem('@Ready4Hire:refresh_token');
        if (!storedRefreshToken) {
          throw new Error('No refresh token available');
        }
        this.refreshToken = storedRefreshToken;
      }
    }

    try {
      // Call refresh endpoint
      const response = await axios.post<{
        access_token: string;
        refresh_token: string;
        token_type: string;
        expires_in: number;
      }>(
        `${API_BASE_URL}/api/${API_VERSION}/auth/refresh`,
        {
          refresh_token: this.refreshToken,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      return {
        access_token: response.data.access_token,
        refresh_token: response.data.refresh_token,
      };
    } catch (error) {
      console.error('Failed to refresh token:', error);
      throw error;
    }
  }

  /**
   * Process queued requests after token refresh
   */
  private processQueue(error: any, token: string | null): void {
    this.failedQueue.forEach((promise) => {
      if (error) {
        promise.reject(error);
      } else {
        promise.resolve(token);
      }
    });
    this.failedQueue = [];
  }

  /**
   * GET request
   */
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  /**
   * POST request
   */
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  /**
   * PUT request
   */
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  /**
   * DELETE request
   */
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Load token on initialization (async, no await to avoid blocking)
apiClient.loadToken().catch((error) => {
  console.error('Failed to load token on initialization:', error);
});

