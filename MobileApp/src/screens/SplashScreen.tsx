/**
 * Splash Screen
 * Shown during app initialization
 */

import React from 'react';
import {View, StyleSheet, ActivityIndicator} from 'react-native';
import {Text} from 'react-native-paper';
import {theme} from '../theme';

const SplashScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Ready4Hire</Text>
      <Text style={styles.subtitle}>AI-Powered Interview Platform</Text>
      <ActivityIndicator
        size="large"
        color={theme.colors.primary}
        style={styles.loader}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
  },
  title: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 32,
  },
  loader: {
    marginTop: 32,
  },
});

export default SplashScreen;

