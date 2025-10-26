/**
 * Home Screen
 * Pantalla principal con dashboard
 */

import React from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Text, Card, Button, Avatar, Title } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useAuth } from '../../store/AuthContext';

const HomeScreen = ({ navigation }: any) => {
  const { user } = useAuth();

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title style={styles.title}>¡Bienvenido, {user?.name}!</Title>
        <Text style={styles.subtitle}>¿Listo para tu próxima entrevista?</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Icon name="trophy" size={32} color="#f59e0b" />
              <Text style={styles.statValue}>{user?.level || 0}</Text>
              <Text style={styles.statLabel}>Nivel</Text>
            </View>
            <View style={styles.statItem}>
              <Icon name="star" size={32} color="#6366f1" />
              <Text style={styles.statValue}>{user?.experience || 0}</Text>
              <Text style={styles.statLabel}>XP</Text>
            </View>
            <View style={styles.statItem}>
              <Icon name="medal" size={32} color="#10b981" />
              <Text style={styles.statValue}>{user?.totalPoints || 0}</Text>
              <Text style={styles.statLabel}>Puntos</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Iniciar Entrevista</Title>
          <Text style={styles.cardDescription}>
            Practica para tu próxima entrevista con nuestra IA
          </Text>
          <Button
            mode="contained"
            icon="message-question"
            onPress={() => navigation.navigate('Interview')}
            style={styles.actionButton}
            buttonColor="#6366f1"
          >
            Comenzar Práctica
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Juegos de Entrenamiento</Title>
          <Text style={styles.cardDescription}>
            Mejora tus habilidades con juegos interactivos
          </Text>
          <Button
            mode="contained"
            icon="controller-classic"
            onPress={() => navigation.navigate('Gamification')}
            style={styles.actionButton}
            buttonColor="#10b981"
          >
            Jugar Ahora
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Ver Reportes</Title>
          <Text style={styles.cardDescription}>
            Analiza tu progreso y áreas de mejora
          </Text>
          <Button
            mode="contained"
            icon="chart-line"
            onPress={() => navigation.navigate('Reports')}
            style={styles.actionButton}
            buttonColor="#f59e0b"
          >
            Ver Estadísticas
          </Button>
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
    backgroundColor: '#1e293b',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  cardTitle: {
    fontSize: 20,
    color: '#ffffff',
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 16,
  },
  actionButton: {
    marginTop: 8,
  },
});

export default HomeScreen;

