/**
 * Notification Service
 * Handles push notifications
 */

import PushNotification from 'react-native-push-notification';
import {Platform} from 'react-native';

class NotificationService {
  private initialized = false;

  /**
   * Initialize notification service
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    PushNotification.configure({
      onRegister: (token) => {
        console.log('📱 Push notification token:', token);
      },
      onNotification: (notification) => {
        console.log('📬 Notification:', notification);
        if (notification.userInteraction) {
          // Handle notification tap
        }
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: Platform.OS === 'ios',
    });

    this.initialized = true;
    console.log('✅ Notification service initialized');
  }

  /**
   * Show local notification
   */
  showNotification(title: string, message: string, data?: any): void {
    PushNotification.localNotification({
      title,
      message,
      data,
      soundName: 'default',
      playSound: true,
    });
  }

  /**
   * Schedule notification
   */
  scheduleNotification(title: string, message: string, date: Date): void {
    PushNotification.localNotificationSchedule({
      title,
      message,
      date,
      soundName: 'default',
      playSound: true,
    });
  }

  /**
   * Cancel all notifications
   */
  cancelAll(): void {
    PushNotification.cancelAllLocalNotifications();
  }
}

export const notificationService = new NotificationService();

