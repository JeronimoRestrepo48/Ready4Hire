/**
 * Main Navigator
 * Handles main app screens after authentication
 */

import React from 'react';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {createStackNavigator} from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import {RootStackParamList} from '../types';
import {theme} from '../theme';

// Screens
import HomeScreen from '../screens/home/HomeScreen';
import InterviewScreen from '../screens/interview/InterviewScreen';
import InterviewListScreen from '../screens/interview/InterviewListScreen';
import GamificationScreen from '../screens/gamification/GamificationScreen';
import LeaderboardScreen from '../screens/gamification/LeaderboardScreen';
import GamesScreen from '../screens/gamification/GamesScreen';
import BadgesScreen from '../screens/gamification/BadgesScreen';
import GameSessionScreen from '../screens/gamification/GameSessionScreen';
import ProfileScreen from '../screens/profile/ProfileScreen';
import SettingsScreen from '../screens/settings/SettingsScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator<RootStackParamList>();

// Home Stack
const HomeStack: React.FC = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{
          headerTitle: 'Ready4Hire',
          headerStyle: {
            backgroundColor: theme.colors.primary,
          },
          headerTintColor: '#fff',
        }}
      />
    </Stack.Navigator>
  );
};

// Interview Stack
const InterviewStack: React.FC = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="InterviewList"
        component={InterviewListScreen}
        options={{
          headerTitle: 'Mis Entrevistas',
        }}
      />
      <Stack.Screen
        name="Interview"
        component={InterviewScreen}
        options={{
          headerTitle: 'Entrevista',
          headerBackTitle: 'Atrás',
        }}
      />
    </Stack.Navigator>
  );
};

// Gamification Stack
const GamificationStack: React.FC = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="Gamification"
        component={GamificationScreen}
        options={{
          headerTitle: 'Gamificación',
        }}
      />
      <Stack.Screen
        name="Leaderboard"
        component={LeaderboardScreen}
        options={{
          headerTitle: 'Ranking Global',
          headerBackTitle: 'Atrás',
        }}
      />
      <Stack.Screen
        name="Games"
        component={GamesScreen}
        options={{
          headerTitle: 'Juegos',
          headerBackTitle: 'Atrás',
        }}
      />
      <Stack.Screen
        name="Badges"
        component={BadgesScreen}
        options={{
          headerTitle: 'Badges',
          headerBackTitle: 'Atrás',
        }}
      />
      <Stack.Screen
        name="GameSession"
        component={GameSessionScreen}
        options={{
          headerTitle: 'Jugando...',
          headerBackTitle: 'Atrás',
        }}
      />
    </Stack.Navigator>
  );
};

// Profile Stack
const ProfileStack: React.FC = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          headerTitle: 'Mi Perfil',
        }}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          headerTitle: 'Configuración',
          headerBackTitle: 'Atrás',
        }}
      />
    </Stack.Navigator>
  );
};

// Main Tab Navigator
const MainNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.disabled,
        headerShown: false,
        tabBarStyle: {
          paddingBottom: 8,
          paddingTop: 8,
          height: 60,
        },
      }}>
      <Tab.Screen
        name="Home"
        component={HomeStack}
        options={{
          tabBarIcon: ({color, size}) => <Icon name="home" size={size} color={color} />,
          tabBarLabel: 'Inicio',
        }}
      />
      <Tab.Screen
        name="Interview"
        component={InterviewStack}
        options={{
          tabBarIcon: ({color, size}) => <Icon name="chat-question" size={size} color={color} />,
          tabBarLabel: 'Entrevistas',
        }}
      />
      <Tab.Screen
        name="Gamification"
        component={GamificationStack}
        options={{
          tabBarIcon: ({color, size}) => <Icon name="trophy" size={size} color={color} />,
          tabBarLabel: 'Gamificación',
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileStack}
        options={{
          tabBarIcon: ({color, size}) => <Icon name="account" size={size} color={color} />,
          tabBarLabel: 'Perfil',
        }}
      />
    </Tab.Navigator>
  );
};

export default MainNavigator;

