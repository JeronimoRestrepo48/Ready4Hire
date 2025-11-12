/**
 * Offline Service
 * Handles offline functionality and sync
 */

import NetInfo from '@react-native-community/netinfo';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Simple EventEmitter implementation for React Native
class EventEmitter {
  private events: {[key: string]: Function[]} = {};

  on(event: string, callback: Function): void {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  off(event: string, callback: Function): void {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }

  emit(event: string, ...args: any[]): void {
    if (this.events[event]) {
      this.events[event].forEach(callback => {
        try {
          callback(...args);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  removeAllListeners(event?: string): void {
    if (event) {
      delete this.events[event];
    } else {
      this.events = {};
    }
  }
}

class OfflineService extends EventEmitter {
  private isOnline: boolean = true;
  private pendingRequests: any[] = [];

  /**
   * Initialize offline service
   */
  async initialize(): Promise<void> {
    // Check initial network state
    const state = await NetInfo.fetch();
    this.isOnline = state.isConnected || false;

    // Listen for network state changes
    NetInfo.addEventListener((state) => {
      const wasOnline = this.isOnline;
      this.isOnline = state.isConnected || false;

      if (!wasOnline && this.isOnline) {
        console.log('🌐 Back online - syncing data');
        this.emit('online');
        this.syncPendingRequests();
      } else if (wasOnline && !this.isOnline) {
        console.log('📴 Went offline');
        this.emit('offline');
      }
    });
  }

  /**
   * Check if currently online
   */
  isConnected(): boolean {
    return this.isOnline;
  }

  /**
   * Queue request for later sync
   */
  queueRequest(request: any): void {
    this.pendingRequests.push({
      ...request,
      timestamp: Date.now(),
    });

    AsyncStorage.setItem(
      '@Ready4Hire:pending_requests',
      JSON.stringify(this.pendingRequests)
    );
  }

  /**
   * Get pending requests
   */
  getPendingRequests(): any[] {
    return this.pendingRequests;
  }

  /**
   * Clear pending requests
   */
  clearPendingRequests(): void {
    this.pendingRequests = [];
    AsyncStorage.removeItem('@Ready4Hire:pending_requests');
  }

  /**
   * Sync pending requests
   */
  private async syncPendingRequests(): Promise<void> {
    if (this.pendingRequests.length === 0) return;

    console.log(`🔄 Syncing ${this.pendingRequests.length} pending requests`);

    try {
      // Load requests from storage
      const stored = await AsyncStorage.getItem('@Ready4Hire:pending_requests');
      if (stored) {
        this.pendingRequests = JSON.parse(stored);
      }

      // Retry each request
      for (const request of this.pendingRequests) {
        try {
          // Re-attempt request
          this.emit('syncRequest', request);
        } catch (error) {
          console.error('Failed to sync request:', error);
        }
      }

      // Clear synced requests
      this.clearPendingRequests();
    } catch (error) {
      console.error('Sync error:', error);
    }
  }
}

export const offlineService = new OfflineService();

