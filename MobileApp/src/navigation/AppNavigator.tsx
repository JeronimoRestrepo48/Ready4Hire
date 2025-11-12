/**
 * App Navigator
 * Main navigation setup for the app
 */

import React, {useEffect} from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createStackNavigator} from '@react-navigation/stack';
import {useSelector, useDispatch} from 'react-redux';
import {RootState} from '../store';
import {loadUserFromStorage} from '../store/slices/authSlice';

// Screens
import AuthNavigator from './AuthNavigator';
import MainNavigator from './MainNavigator';
import SplashScreen from '../screens/SplashScreen';

const Stack = createStackNavigator();

const AppNavigator: React.FC = () => {
  const dispatch = useDispatch();
  const {isAuthenticated, loading} = useSelector((state: RootState) => state.auth);

  useEffect(() => {
    // Load user from AsyncStorage on app start
    try {
      dispatch(loadUserFromStorage() as any);
    } catch (error) {
      console.error('Failed to load user from storage:', error);
    }
  }, [dispatch]);

  if (loading) {
    return <SplashScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{headerShown: false}}>
        {isAuthenticated ? (
          <Stack.Screen name="Main" component={MainNavigator} />
        ) : (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;

