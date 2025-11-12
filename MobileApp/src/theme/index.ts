import {MD3LightTheme} from 'react-native-paper';

export const theme = {
  ...MD3LightTheme,
  colors: {
    ...MD3LightTheme.colors,
    primary: '#6366F1', // Indigo
    secondary: '#8B5CF6', // Purple
    tertiary: '#EC4899', // Pink
    accent: '#10B981', // Green
    error: '#EF4444', // Red
    warning: '#F59E0B', // Amber
    success: '#10B981', // Green
    background: '#F9FAFB',
    surface: '#FFFFFF',
    text: '#1F2937',
    disabled: '#9CA3AF',
    placeholder: '#6B7280',
    backdrop: 'rgba(0, 0, 0, 0.5)',
    
    // Gamification colors
    badge: {
      common: '#60A5FA', // Blue
      rare: '#A78BFA', // Purple
      epic: '#F87171', // Orange
      legendary: '#FBBF24', // Gold
    },
    
    // Status colors
    interview: {
      active: '#6366F1',
      completed: '#10B981',
      paused: '#F59E0B',
    },
  },
  
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
  
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
  },
};

export type Theme = typeof theme;

