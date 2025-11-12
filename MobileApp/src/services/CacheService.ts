/**
 * Cache Service
 * Handles caching of API responses and offline data
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
}

class CacheService {
  private prefix = '@Ready4Hire:cache:';
  private defaultTTL = 3600000; // 1 hour

  /**
   * Initialize cache service
   */
  async initialize(): Promise<void> {
    // Clean expired entries
    await this.cleanExpired();
  }

  /**
   * Get cached data
   */
  async get<T>(key: string): Promise<T | null> {
    try {
      const cached = await AsyncStorage.getItem(this.prefix + key);
      if (!cached) return null;

      const entry: CacheEntry<T> = JSON.parse(cached);

      // Check if expired
      if (Date.now() - entry.timestamp > entry.ttl) {
        await this.delete(key);
        return null;
      }

      return entry.data;
    } catch (error) {
      console.error('Cache get error:', error);
      return null;
    }
  }

  /**
   * Set cached data
   */
  async set<T>(key: string, data: T, ttl?: number): Promise<void> {
    try {
      const entry: CacheEntry<T> = {
        data,
        timestamp: Date.now(),
        ttl: ttl || this.defaultTTL,
      };
      await AsyncStorage.setItem(this.prefix + key, JSON.stringify(entry));
    } catch (error) {
      console.error('Cache set error:', error);
    }
  }

  /**
   * Delete cached data
   */
  async delete(key: string): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.prefix + key);
    } catch (error) {
      console.error('Cache delete error:', error);
    }
  }

  /**
   * Clear all cached data
   */
  async clear(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter((key) => key.startsWith(this.prefix));
      await AsyncStorage.multiRemove(cacheKeys);
    } catch (error) {
      console.error('Cache clear error:', error);
    }
  }

  /**
   * Clean expired entries
   */
  private async cleanExpired(): Promise<void> {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const cacheKeys = keys.filter((key) => key.startsWith(this.prefix));

      const now = Date.now();
      const toRemove: string[] = [];

      for (const key of cacheKeys) {
        const cached = await AsyncStorage.getItem(key);
        if (!cached) continue;

        const entry: CacheEntry<any> = JSON.parse(cached);
        if (now - entry.timestamp > entry.ttl) {
          toRemove.push(key);
        }
      }

      if (toRemove.length > 0) {
        await AsyncStorage.multiRemove(toRemove);
        console.log(`🧹 Cleaned ${toRemove.length} expired cache entries`);
      }
    } catch (error) {
      console.error('Cache clean error:', error);
    }
  }
}

export const cacheService = new CacheService();

