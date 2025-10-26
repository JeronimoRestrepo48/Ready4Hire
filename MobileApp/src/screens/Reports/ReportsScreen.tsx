/**
 * Reports Screen
 * Pantalla de reportes y estadísticas
 */

import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, Dimensions } from 'react-native';
import { Text, Card, Title, Chip, Divider } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { LineChart } from 'react-native-chart-kit';
import { useAuth } from '../../store/AuthContext';

const screenWidth = Dimensions.get('window').width;

const ReportsScreen = () => {
  const { user } = useAuth();
  const [reports, setReports] = useState([]);
  const [selectedFilter, setSelectedFilter] = useState('all');

  useEffect(() => {
    loadReports();
  }, [selectedFilter]);

  const loadReports = async () => {
    try {
      // TODO: Llamar a GraphQL o REST API
      // const response = await apolloClient.query({
      //   query: GET_USER_REPORTS,
      //   variables: { userId: user?.id }
      // });
      // setReports(response.data.reports);
    } catch (error) {
      console.error('Error loading reports:', error);
    }
  };

  const chartData = {
    labels: ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4'],
    datasets: [
      {
        data: [65, 72, 78, 85],
        color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
        strokeWidth: 3,
      },
    ],
  };

  const chartConfig = {
    backgroundColor: '#1e293b',
    backgroundGradientFrom: '#1e293b',
    backgroundGradientTo: '#0f172a',
    decimalPlaces: 0,
    color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(148, 163, 184, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#6366f1',
    },
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title style={styles.title}>Mis Reportes</Title>
        <Text style={styles.subtitle}>Analiza tu progreso</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Resumen General</Title>
          <View style={styles.summaryRow}>
            <View style={styles.summaryBox}>
              <Icon name="checkbox-marked-circle" size={32} color="#10b981" />
              <Text style={styles.summaryValue}>12</Text>
              <Text style={styles.summaryLabel}>Completadas</Text>
            </View>
            <View style={styles.summaryBox}>
              <Icon name="chart-line" size={32} color="#6366f1" />
              <Text style={styles.summaryValue}>78%</Text>
              <Text style={styles.summaryLabel}>Promedio</Text>
            </View>
            <View style={styles.summaryBox}>
              <Icon name="trending-up" size={32} color="#f59e0b" />
              <Text style={styles.summaryValue}>+15%</Text>
              <Text style={styles.summaryLabel}>Mejora</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Evolución de Puntuación</Title>
          <LineChart
            data={chartData}
            width={screenWidth - 64}
            height={220}
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
          />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Fortalezas</Title>
          <View style={styles.chipContainer}>
            <Chip icon="check-circle" style={styles.strengthChip}>
              Algoritmos
            </Chip>
            <Chip icon="check-circle" style={styles.strengthChip}>
              Estructuras de Datos
            </Chip>
            <Chip icon="check-circle" style={styles.strengthChip}>
              Design Patterns
            </Chip>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Áreas de Mejora</Title>
          <View style={styles.chipContainer}>
            <Chip icon="alert-circle" style={styles.improvementChip}>
              Concurrencia
            </Chip>
            <Chip icon="alert-circle" style={styles.improvementChip}>
              Redes
            </Chip>
            <Chip icon="alert-circle" style={styles.improvementChip}>
              Seguridad
            </Chip>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.cardTitle}>Historial de Entrevistas</Title>
          {reports.length === 0 ? (
            <Text style={styles.emptyText}>
              No hay reportes disponibles. Completa tu primera entrevista!
            </Text>
          ) : (
            reports.map((report: any, index: number) => (
              <View key={index}>
                <View style={styles.reportRow}>
                  <View style={styles.reportInfo}>
                    <Text style={styles.reportRole}>{report.role}</Text>
                    <Text style={styles.reportDate}>{report.date}</Text>
                  </View>
                  <View style={styles.reportScore}>
                    <Text style={styles.scoreValue}>{report.score}%</Text>
                    <Icon 
                      name={report.score >= 70 ? 'trending-up' : 'trending-down'} 
                      size={20} 
                      color={report.score >= 70 ? '#10b981' : '#ef4444'} 
                    />
                  </View>
                </View>
                {index < reports.length - 1 && <Divider style={styles.divider} />}
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
  cardTitle: {
    fontSize: 20,
    color: '#ffffff',
    marginBottom: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  summaryBox: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 8,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  strengthChip: {
    backgroundColor: '#10b98120',
    borderColor: '#10b981',
    borderWidth: 1,
  },
  improvementChip: {
    backgroundColor: '#ef444420',
    borderColor: '#ef4444',
    borderWidth: 1,
  },
  emptyText: {
    color: '#94a3b8',
    textAlign: 'center',
    padding: 16,
    fontStyle: 'italic',
  },
  reportRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  reportInfo: {
    flex: 1,
  },
  reportRole: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  reportDate: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  reportScore: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  scoreValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#6366f1',
  },
  divider: {
    backgroundColor: '#334155',
  },
});

export default ReportsScreen;

