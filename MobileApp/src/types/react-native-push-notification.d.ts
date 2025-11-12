declare module 'react-native-push-notification' {
  interface PushNotificationOptions {
    onRegister?: (token: any) => void;
    onNotification?: (notification: any) => void;
    permissions?: {
      alert?: boolean;
      badge?: boolean;
      sound?: boolean;
    };
    popInitialNotification?: boolean;
    requestPermissions?: boolean;
  }

  interface LocalNotification {
    title: string;
    message: string;
    data?: any;
    soundName?: string;
    playSound?: boolean;
    date?: Date;
  }

  class PushNotification {
    static configure(options: PushNotificationOptions): void;
    static localNotification(notification: LocalNotification): void;
    static localNotificationSchedule(notification: LocalNotification): void;
    static cancelAllLocalNotifications(): void;
  }

  export default PushNotification;
}

