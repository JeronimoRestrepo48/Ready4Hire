/**
 * React Native Configuration
 * 
 * Temporarily excludes react-native-code-push from auto-linking
 * due to Gradle compatibility issues with Gradle 7.6.3 + AGP 7.4.2
 */

module.exports = {
  dependencies: {
    'react-native-code-push': {
      platforms: {
        android: null, // Disable Android platform, auto-linking will skip it
        ios: null, // Disable iOS platform
      },
    },
  },
  assets: ['./src/assets/fonts/', './node_modules/react-native-vector-icons/Fonts/'],
};

