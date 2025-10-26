/**
 * Gamification Screen
 * Pantalla de juegos y gamificación
 */

import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, FlatList } from 'react-native';
import { Text, Card, Button, Title, Badge, ProgressBar, Avatar } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useAuth } from '../../store/AuthContext';
import axios from 'axios';

const GAMES = [
  {
    id: 'quick_quiz',
    name: 'Quick Quiz',
    description: 'Responde preguntas de opción múltiple',
    icon: 'lightning-bolt',
    color: '#f59e0b',
    xpReward: 50,
  },
  {
    id: 'code_challenge',
    name: 'Code Challenge',
    description: 'Resuelve problemas de programación',
    icon: 'code-braces',
    color: '#6366f1',
    xpReward: 100,
  },
  {
    id: 'speed_round',
    name: 'Speed Round',
    description: 'Responde lo más rápido posible',
    icon: 'timer',
    color: '#ef4444',
    xpReward: 75,
  },
  {
    id: 'scenario_simulator',
    name: 'Scenario Simulator',
    description: 'Toma decisiones en situaciones reales',
    icon: 'thought-bubble',
    color: '#10b981',
    xpReward: 120,
  },
];

const GamificationScreen = ({ navigation }: any) => {
  const { user } = useAuth();
  const [badges, setBadges] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);

  useEffect(() => {
    loadGamificationData();
  }, []);

  const loadGamificationData = async () => {
    try {
      // TODO: Llamar a GraphQL o REST API
      // const response = await axios.get(`${API_URL}/gamification/badges/${user?.id}`);
      // setBadges(response.data);
    } catch (error) {
      console.error('Error loading gamification data:', error);
    }
  };

  const renderGameCard = ({ item }: any) => (
    <Card style={[styles.gameCard, { borderLeftColor: item.color }]}>
      <Card.Content>
        <View style={styles.gameHeader}>
          <Icon name={item.icon} size={40} color={item.color} />
          <View style={styles.gameInfo}>
            <Title style={styles.gameName}>{item.name}</Title>
            <Text style={styles.gameDescription}>{item.description}</Text>
          </View>
        </View>
        <View style={styles.gameFooter}>
          <View style={styles.xpBadge}>
            <Icon name="star" size={16} color="#f59e0b" />
            <Text style={styles.xpText}>+{item.xpReward} XP</Text>
          </View>
          <Button
            mode="contained"
            compact
            onPress={() => console.log(`Playing ${item.id}`)}
            buttonColor={item.color}
          >
            Jugar
          </Button>
        </View>
      </Card.Content>
    </Card>
  );

  const xpForNextLevel = user ? 100 * Math.pow(user.level + 1, 2) : 100;
  const levelProgress = user ? user.experience / xpForNextLevel : 0;

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title style={styles.title}>Gamificación</Title>
        <Text style={styles.subtitle}>Entrena y gana recompensas</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.statsRow}>
            <View style={styles.statBox}>
              <Icon name="trophy" size={32} color="#f59e0b" />
              <Text style={styles.statValue}>Nivel {user?.level || 1}</Text>
            </View>
            <View style={styles.statBox}>
              <Icon name="star" size={32} color="#6366f1" />
              <Text style={styles.statValue}>{user?.experience || 0} XP</Text>
            </View>
            <View style={styles.statBox}>
              <Icon name="fire" size={32} color="#ef4444" />
              <Text style={styles.statValue}>{user?.streakDays || 0} días</Text>
            </View>
          </View>
          <View style={styles.progressContainer}>
            <Text style={styles.progressText}>
              Progreso al nivel {(user?.level || 0) + 1}
            </Text>
            <ProgressBar 
              progress={levelProgress} 
              color="#6366f1" 
              style={styles.progressBar}
            />
            <Text style={styles.progressDetail}>
              {user?.experience || 0} / {xpForNextLevel} XP
            </Text>
          </View>
        </Card.Content>
      </Card>

      <View style={styles.section}>
        <Title style={styles.sectionTitle}>Juegos Disponibles</Title>
        <FlatList
          data={GAMES}
          renderItem={renderGameCard}
          keyExtractor={item => item.id}
          scrollEnabled={false}
        />
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Top 5 Leaderboard</Title>
          {leaderboard.length === 0 ? (
            <Text style={styles.emptyText}>Cargando tabla de líderes...</Text>
          ) : (
            leaderboard.map((leader: any, index: number) => (
              <View key={index} style={styles.leaderRow}>
                <Text style={styles.leaderRank}>#{index + 1}</Text>
                <Avatar.Text 
                  size={32} 
                  label={leader.name[0]} 
                  style={styles.avatar}
                />
                <View style={styles.leaderInfo}>
                  <Text style={styles.leaderName}>{leader.name}</Text>
                  <Text style={styles.leaderPoints}>{leader.points} pts</Text>
                </View>
              </View>
            ))
          )}
        </Card.Content>
      </Card>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0e1a',
  },
  header: {
    padding: 24,
    paddingTop: 48,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  subtitle: {
    fontSize: 16,
    color: '#94a3b8',
    marginTop: 4,
  },
  card: {
    margin: 16,
    marginTop: 0,
    marginBottom: 16,
    backgroundColor: '#1e293b',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
  },
  statBox: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 4,
  },
  progressContainer: {
    marginTop: 8,
  },
  progressText: {
    color: '#94a3b8',
    fontSize: 14,
    marginBottom: 8,
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  progressDetail: {
    color: '#94a3b8',
    fontSize: 12,
    marginTop: 4,
    textAlign: 'right',
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 20,
    color: '#ffffff',
    marginBottom: 12,
  },
  gameCard: {
    backgroundColor: '#1e293b',
    marginBottom: 12,
    borderLeftWidth: 4,
  },
  gameHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  gameInfo: {
    marginLeft: 16,
    flex: 1,
  },
  gameName: {
    fontSize: 18,
    color: '#ffffff',
    marginBottom: 4,
  },
  gameDescription: {
    fontSize: 14,
    color: '#94a3b8',
  },
  gameFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  xpBadge: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  xpText: {
    color: '#f59e0b',
    fontWeight: 'bold',
    marginLeft: 4,
  },
  emptyText: {
    color: '#94a3b8',
    textAlign: 'center',
    padding: 16,
  },
  leaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  leaderRank: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#f59e0b',
    width: 40,
  },
  avatar: {
    marginRight: 12,
  },
  leaderInfo: {
    flex: 1,
  },
  leaderName: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: 'bold',
  },
  leaderPoints: {
    fontSize: 14,
    color: '#94a3b8',
  },
});

export default GamificationScreen;

