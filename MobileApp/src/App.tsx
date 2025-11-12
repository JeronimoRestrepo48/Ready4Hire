/**
 * Ready4Hire Mobile App
 * 
 * Main App Component with Navigation and Redux Store
 */

import React, {useEffect} from 'react';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import {PaperProvider} from 'react-native-paper';
import {Provider as ReduxProvider} from 'react-redux';
import {GestureHandlerRootView} from 'react-native-gesture-handler';
import {StatusBar, Platform} from 'react-native';
// Temporarily disabled CodePush due to Gradle compatibility issues
// import CodePush from 'react-native-code-push';

import {store} from './store';
import AppNavigator from './navigation/AppNavigator';
import {theme} from './theme';
import {initializeServices} from './services/ServiceInitializer';

// Configure react-native-vector-icons for react-native-paper
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';

// Ensure fonts are loaded
MaterialCommunityIcons.loadFont().catch((err: any) => {
  console.warn('Failed to load MaterialCommunityIcons font:', err);
});

const App: React.FC = () => {
  useEffect(() => {
    // Initialize services (API, notifications, etc.)
    initializeServices().catch((error) => {
      console.error('Failed to initialize services:', error);
    });
  }, []);

  return (
    <SafeAreaProvider>
      <ReduxProvider store={store}>
        <PaperProvider theme={theme} settings={{
          icon: (props) => <MaterialCommunityIcons name={props.name} size={props.size} color={props.color} />,
        }}>
          <GestureHandlerRootView style={{flex: 1}}>
            <StatusBar
              barStyle={Platform.OS === 'ios' ? 'dark-content' : 'light-content'}
              backgroundColor={theme.colors.primary}
            />
            <AppNavigator />
          </GestureHandlerRootView>
        </PaperProvider>
      </ReduxProvider>
    </SafeAreaProvider>
  );
};

// Temporarily disabled CodePush due to Gradle compatibility issues
// Wrap with CodePush for OTA updates (optional)
// export default CodePush({
//   checkFrequency: CodePush.CheckFrequency.ON_APP_RESUME,
// })(App);

// Export App directly without CodePush wrapper
export default App;

