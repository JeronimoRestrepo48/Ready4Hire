/**
 * Tab Navigator
 * Bottom tabs para navegación principal
 */

import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

// Placeholders - implementar después
import HomeScreen from '../screens/Home/HomeScreen';
import InterviewScreen from '../screens/Interview/InterviewScreen';
import GamificationScreen from '../screens/Gamification/GamificationScreen';
import ReportsScreen from '../screens/Reports/ReportsScreen';
import ProfileScreen from '../screens/Profile/ProfileScreen';

const Tab = createBottomTabNavigator();

const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#1e293b',
          borderTopColor: '#334155',
          borderTopWidth: 1,
          height: 60,
          paddingBottom: 8,
        },
        tabBarActiveTintColor: '#6366f1',
        tabBarInactiveTintColor: '#94a3b8',
        tabBarIcon: ({ color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'Home':
              iconName = 'home';
              break;
            case 'Interview':
              iconName = 'message-question';
              break;
            case 'Gamification':
              iconName = 'trophy';
              break;
            case 'Reports':
              iconName = 'chart-line';
              break;
            case 'Profile':
              iconName = 'account';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{ tabBarLabel: 'Inicio' }}
      />
      <Tab.Screen 
        name="Interview" 
        component={InterviewScreen}
        options={{ tabBarLabel: 'Entrevista' }}
      />
      <Tab.Screen 
        name="Gamification" 
        component={GamificationScreen}
        options={{ tabBarLabel: 'Juegos' }}
      />
      <Tab.Screen 
        name="Reports" 
        component={ReportsScreen}
        options={{ tabBarLabel: 'Reportes' }}
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{ tabBarLabel: 'Perfil' }}
      />
    </Tab.Navigator>
  );
};

export default TabNavigator;

