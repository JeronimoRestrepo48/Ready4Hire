/**
 * Service Initializer
 * Initializes all app services on startup
 */

import {apiClient} from './api/ApiClient';
import {notificationService} from './NotificationService';
import {cacheService} from './CacheService';
import {offlineService} from './OfflineService';

export async function initializeServices(): Promise<void> {
  try {
    console.log('🚀 Initializing services...');

    // Initialize API Client
    await apiClient.loadToken();
    console.log('✅ API Client initialized');

    // Initialize Cache
    await cacheService.initialize();
    console.log('✅ Cache Service initialized');

    // Initialize Offline Service
    await offlineService.initialize();
    console.log('✅ Offline Service initialized');

    // Initialize Notification Service (only if enabled)
    const enableNotifications = process.env.ENABLE_PUSH_NOTIFICATIONS === 'true';
    if (enableNotifications) {
      await notificationService.initialize();
      console.log('✅ Notification Service initialized');
    }

    console.log('✅ All services initialized successfully');
  } catch (error) {
    console.error('❌ Failed to initialize services:', error);
  }
}

