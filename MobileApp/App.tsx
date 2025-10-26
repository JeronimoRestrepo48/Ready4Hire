/**
 * Ready4Hire Mobile App
 * Main entry point
 */

import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ApolloProvider } from '@apollo/client';
import { NavigationContainer } from '@react-navigation/native';
import { PaperProvider, MD3DarkTheme, MD3LightTheme } from 'react-native-paper';
import { useColorScheme } from 'react-native';

import { apolloClient } from './src/services/api/apolloClient';
import { AuthProvider } from './src/store/AuthContext';
import { InterviewProvider } from './src/store/InterviewContext';
import AppNavigator from './src/navigation/AppNavigator';

const App = (): JSX.Element => {
  const colorScheme = useColorScheme();
  const theme = colorScheme === 'dark' ? MD3DarkTheme : MD3LightTheme;

  // Personalizar tema
  const customTheme = {
    ...theme,
    colors: {
      ...theme.colors,
      primary: '#6366f1',
      secondary: '#10b981',
      tertiary: '#f59e0b',
    },
  };

  return (
    <SafeAreaProvider>
      <ApolloProvider client={apolloClient}>
        <PaperProvider theme={customTheme}>
          <AuthProvider>
            <InterviewProvider>
              <NavigationContainer>
                <AppNavigator />
              </NavigationContainer>
            </InterviewProvider>
          </AuthProvider>
        </PaperProvider>
      </ApolloProvider>
    </SafeAreaProvider>
  );
};

export default App;

