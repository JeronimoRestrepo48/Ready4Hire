# 📱 Ready4Hire Mobile App

Mobile application for Ready4Hire technical interview platform.

## 🚀 Features

- ✅ AI-Powered Technical Interviews
- 🎮 Gamification System (Badges, Levels, Points)
- 📊 Real-time Interview Feedback
- 🏆 Leaderboard & Achievements
- 👤 User Profile & Settings
- 🔔 Push Notifications
- 💾 Offline Mode
- 🌐 Multi-language Support

## 📋 Prerequisites

- Node.js >= 18
- React Native CLI
- Xcode (for iOS)
- Android Studio (for Android)
- Watchman (recommended)

## 🛠️ Installation

```bash
# Install dependencies
npm install

# iOS (macOS only)
cd ios && pod install && cd ..

# Start Metro bundler
npm start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

## 🏗️ Project Structure

```
MobileApp/
├── src/
│   ├── App.tsx                 # Main app component
│   ├── components/             # Reusable UI components
│   ├── screens/               # Screen components
│   ├── navigation/            # Navigation setup
│   ├── services/              # API services
│   ├── store/                 # Redux store
│   ├── types/                 # TypeScript types
│   ├── utils/                 # Utilities
│   └── assets/               # Images, fonts, etc.
├── android/                   # Android native code
├── ios/                       # iOS native code
└── package.json
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```env
API_BASE_URL=https://api.ready4hire.com
```

## 📱 Build

### Android
```bash
cd android
./gradlew assembleRelease
```

### iOS
```bash
cd ios
xcodebuild -workspace Ready4Hire.xcworkspace -scheme Ready4Hire -configuration Release
```

## 🧪 Testing

```bash
npm test
```

## 📄 License

MIT

## 👥 Team

- Ready4Hire Development Team

