/**
 * Home Screen
 * Main dashboard for the app
 */

import React, {useEffect} from 'react';
import {View, StyleSheet, ScrollView, TouchableOpacity} from 'react-native';
import {Card, Text, Avatar, Button} from 'react-native-paper';
import {useSelector, useDispatch} from 'react-redux';
import {RootState, AppDispatch} from '../../store';
import {fetchUserStats} from '../../store/slices/gamificationSlice';
import {theme} from '../../theme';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import {HomeScreenProps} from '../../types/navigation';

const HomeScreen: React.FC<HomeScreenProps> = ({navigation}) => {
  const dispatch = useDispatch<AppDispatch>();
  const {user} = useSelector((state: RootState) => state.auth);
  const {stats} = useSelector((state: RootState) => state.gamification);

  useEffect(() => {
    if (user?.id) {
      dispatch(fetchUserStats(user.id));
    }
  }, [user, dispatch]);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.welcomeSection}>
          <Text style={styles.greeting}>¡Hola, {user?.name || 'Usuario'}! 👋</Text>
          <Text style={styles.subtitle}>
            {stats?.level ? `Nivel ${stats.level} • ${stats.totalPoints} pts` : 'Bienvenido'}
          </Text>
        </View>
        <Avatar.Image size={64} source={{uri: user?.avatarUrl}} />
      </View>

      {/* Quick Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Acciones Rápidas</Text>
        <View style={styles.quickActions}>
          <TouchableOpacity
            style={[styles.quickAction, {backgroundColor: theme.colors.primary}]}
            onPress={() => (navigation as any).navigate('Interview', {screen: 'InterviewList'})}>
            <Icon name="chat-question" size={32} color="#fff" />
            <Text style={styles.quickActionText}>Iniciar Entrevista</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.quickAction, {backgroundColor: theme.colors.secondary}]}
            onPress={() => (navigation as any).navigate('Gamification', {screen: 'Games'})}>
            <Icon name="puzzle" size={32} color="#fff" />
            <Text style={styles.quickActionText}>Juegos</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Stats Cards */}
      {stats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Tu Progreso</Text>
          <View style={styles.statsGrid}>
            <Card style={styles.statCard}>
              <Card.Content>
                <Text style={styles.statValue}>{stats.level}</Text>
                <Text style={styles.statLabel}>Nivel</Text>
              </Card.Content>
            </Card>

            <Card style={styles.statCard}>
              <Card.Content>
                <Text style={styles.statValue}>{stats.streakDays}</Text>
                <Text style={styles.statLabel}>Días de Racha</Text>
              </Card.Content>
            </Card>

            <Card style={styles.statCard}>
              <Card.Content>
                <Text style={styles.statValue}>{stats.totalGamesWon}</Text>
                <Text style={styles.statLabel}>Juegos Ganados</Text>
              </Card.Content>
            </Card>

            <Card style={styles.statCard}>
              <Card.Content>
                <Text style={styles.statValue}>#{stats.rank}</Text>
                <Text style={styles.statLabel}>Ranking Global</Text>
              </Card.Content>
            </Card>
          </View>
        </View>
      )}

      {/* Recent Interviews */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Entrevistas Recientes</Text>
        <Card style={styles.card}>
          <Card.Content>
            <Text>No hay entrevistas recientes</Text>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('InterviewList')}
              style={styles.button}>
              Ver Todas
            </Button>
          </Card.Content>
        </Card>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
    backgroundColor: theme.colors.primary,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
  },
  welcomeSection: {
    flex: 1,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
  },
  section: {
    padding: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: theme.colors.text,
    marginBottom: 16,
  },
  quickActions: {
    flexDirection: 'row',
    gap: 12,
  },
  quickAction: {
    flex: 1,
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  quickActionText: {
    color: '#fff',
    marginTop: 8,
    fontSize: 14,
    fontWeight: '600',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: theme.colors.surface,
  },
  statValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: theme.colors.primary,
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 14,
    color: theme.colors.text,
    opacity: 0.7,
  },
  card: {
    marginTop: 12,
    backgroundColor: theme.colors.surface,
  },
  button: {
    marginTop: 12,
  },
});

export default HomeScreen;

