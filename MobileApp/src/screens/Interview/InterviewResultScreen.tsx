/**
 * Interview Result Screen
 * Pantalla de resultados al finalizar entrevista
 */

import React, { useEffect, useState } from 'react';
import { View, ScrollView, StyleSheet, Share, Alert } from 'react-native';
import { Text, Card, Button, Divider, ProgressBar, Chip } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { LineChart } from 'react-native-chart-kit';
import { Dimensions } from 'react-native';

const screenWidth = Dimensions.get('window').width;

const InterviewResultScreen = ({ route, navigation }: any) => {
  const { report } = route.params || {};
  const [certificateUrl, setCertificateUrl] = useState<string | null>(null);

  useEffect(() => {
    // Si es modo examen y aprobó, generar certificado
    if (report?.certifiable && report?.averageScore >= 70) {
      // TODO: Llamar a API para generar certificado
      setCertificateUrl('https://example.com/certificate/123');
    }
  }, [report]);

  const handleShare = async () => {
    try {
      await Share.share({
        message: `¡Completé una entrevista en Ready4Hire!\n\nProfesión: ${report?.role}\nPuntuación: ${report?.averageScore}%\nTasa de éxito: ${report?.successRate}%`,
        title: 'Mis resultados en Ready4Hire',
      });
    } catch (error) {
      console.error('Error sharing:', error);
    }
  };

  const handleViewCertificate = () => {
    if (certificateUrl) {
      // TODO: Navegar a pantalla de certificado
      Alert.alert('Certificado', 'Ver certificado en pantalla completa');
    }
  };

  const getPerformanceLevel = (score: number) => {
    if (score >= 90) return { label: 'Excelente', color: '#10b981', icon: 'star' };
    if (score >= 70) return { label: 'Bueno', color: '#6366f1', icon: 'thumb-up' };
    if (score >= 50) return { label: 'Aceptable', color: '#f59e0b', icon: 'alert-circle' };
    return { label: 'Necesita Mejora', color: '#ef4444', icon: 'alert' };
  };

  if (!report) {
    return (
      <View style={styles.centerContainer}>
        <Icon name="alert-circle" size={64} color="#ef4444" />
        <Text style={styles.errorText}>No hay resultados disponibles</Text>
        <Button mode="contained" onPress={() => navigation.navigate('Home')}>
          Volver al inicio
        </Button>
      </View>
    );
  }

  const performance = getPerformanceLevel(report.averageScore);
  const chartData = {
    labels: report.scoreTrend?.map((_: any, i: number) => `P${i + 1}`) || ['P1', 'P2', 'P3'],
    datasets: [{ data: report.scoreTrend || [70, 75, 80] }],
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Icon name="check-circle" size={64} color={performance.color} />
        <Text style={styles.congratsText}>¡Entrevista Completada!</Text>
        <Text style={styles.subtitleText}>Aquí están tus resultados</Text>
      </View>

      {/* Score Card */}
      <Card style={styles.scoreCard}>
        <Card.Content>
          <View style={styles.scoreHeader}>
            <View style={styles.scoreMain}>
              <Text style={[styles.scoreValue, { color: performance.color }]}>
                {Math.round(report.averageScore)}%
              </Text>
              <Text style={styles.scoreLabel}>Puntuación Final</Text>
            </View>
            <Chip
              icon={performance.icon}
              style={[styles.performanceChip, { backgroundColor: `${performance.color}20` }]}
              textStyle={{ color: performance.color }}
            >
              {performance.label}
            </Chip>
          </View>

          <Divider style={styles.divider} />

          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Icon name="check-circle" size={24} color="#10b981" />
              <Text style={styles.statValue}>{report.correctAnswers || 0}</Text>
              <Text style={styles.statLabel}>Correctas</Text>
            </View>
            <View style={styles.statItem}>
              <Icon name="chart-line" size={24} color="#6366f1" />
              <Text style={styles.statValue}>{report.successRate || 0}%</Text>
              <Text style={styles.statLabel}>Tasa Éxito</Text>
            </View>
            <View style={styles.statItem}>
              <Icon name="trophy" size={24} color="#f59e0b" />
              <Text style={styles.statValue}>{report.percentile || 0}</Text>
              <Text style={styles.statLabel}>Percentil</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      {/* Performance Chart */}
      {report.scoreTrend && report.scoreTrend.length > 0 && (
        <Card style={styles.card}>
          <Card.Content>
            <Text style={styles.cardTitle}>Evolución de Puntuación</Text>
            <LineChart
              data={chartData}
              width={screenWidth - 64}
              height={200}
              chartConfig={{
                backgroundColor: '#1e293b',
                backgroundGradientFrom: '#1e293b',
                backgroundGradientTo: '#0f172a',
                decimalPlaces: 0,
                color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(148, 163, 184, ${opacity})`,
                style: { borderRadius: 16 },
                propsForDots: { r: '4', strokeWidth: '2', stroke: '#6366f1' },
              }}
              bezier
              style={styles.chart}
            />
          </Card.Content>
        </Card>
      )}

      {/* Strengths */}
      {report.strengths && report.strengths.length > 0 && (
        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Icon name="check-circle" size={24} color="#10b981" />
              <Text style={styles.cardTitle}>Fortalezas</Text>
            </View>
            <View style={styles.chipContainer}>
              {report.strengths.map((strength: string, index: number) => (
                <Chip key={index} icon="check" style={styles.strengthChip}>
                  {strength}
                </Chip>
              ))}
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Areas of Improvement */}
      {report.improvements && report.improvements.length > 0 && (
        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <Icon name="alert-circle" size={24} color="#f59e0b" />
              <Text style={styles.cardTitle}>Áreas de Mejora</Text>
            </View>
            <View style={styles.chipContainer}>
              {report.improvements.map((improvement: string, index: number) => (
                <Chip key={index} icon="alert" style={styles.improvementChip}>
                  {improvement}
                </Chip>
              ))}
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Certificate */}
      {certificateUrl && (
        <Card style={styles.certificateCard}>
          <Card.Content>
            <View style={styles.certificateHeader}>
              <Icon name="certificate" size={48} color="#f59e0b" />
              <View style={styles.certificateText}>
                <Text style={styles.certificateTitle}>¡Certificado Disponible!</Text>
                <Text style={styles.certificateSubtitle}>
                  Has aprobado el examen con {Math.round(report.averageScore)}%
                </Text>
              </View>
            </View>
            <Button
              mode="contained"
              icon="download"
              onPress={handleViewCertificate}
              buttonColor="#f59e0b"
              style={styles.certificateButton}
            >
              Ver Certificado
            </Button>
          </Card.Content>
        </Card>
      )}

      {/* Actions */}
      <View style={styles.actions}>
        <Button
          mode="outlined"
          icon="share-variant"
          onPress={handleShare}
          style={styles.actionButton}
          textColor="#6366f1"
        >
          Compartir
        </Button>
        <Button
          mode="contained"
          icon="replay"
          onPress={() => navigation.navigate('Interview')}
          style={styles.actionButton}
          buttonColor="#6366f1"
        >
          Nueva Entrevista
        </Button>
      </View>

      <Button
        mode="text"
        onPress={() => navigation.navigate('Home')}
        style={styles.homeButton}
      >
        Volver al Inicio
      </Button>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0e1a',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0e1a',
    padding: 24,
  },
  errorText: {
    fontSize: 18,
    color: '#94a3b8',
    marginVertical: 16,
  },
  header: {
    alignItems: 'center',
    padding: 32,
    paddingTop: 48,
  },
  congratsText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 16,
  },
  subtitleText: {
    fontSize: 16,
    color: '#94a3b8',
    marginTop: 8,
  },
  scoreCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#1e293b',
  },
  scoreHeader: {
    alignItems: 'center',
  },
  scoreMain: {
    alignItems: 'center',
    marginBottom: 16,
  },
  scoreValue: {
    fontSize: 64,
    fontWeight: 'bold',
  },
  scoreLabel: {
    fontSize: 16,
    color: '#94a3b8',
    marginTop: 4,
  },
  performanceChip: {
    marginBottom: 16,
  },
  divider: {
    marginVertical: 16,
    backgroundColor: '#334155',
  },
  statsGrid: {
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
  card: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#1e293b',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginLeft: 8,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
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
    backgroundColor: '#f59e0b20',
    borderColor: '#f59e0b',
    borderWidth: 1,
  },
  certificateCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#f59e0b20',
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  certificateHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  certificateText: {
    marginLeft: 16,
    flex: 1,
  },
  certificateTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  certificateSubtitle: {
    fontSize: 14,
    color: '#fbbf24',
    marginTop: 4,
  },
  certificateButton: {
    marginTop: 8,
  },
  actions: {
    flexDirection: 'row',
    padding: 16,
    gap: 16,
  },
  actionButton: {
    flex: 1,
  },
  homeButton: {
    marginBottom: 32,
  },
});

export default InterviewResultScreen;

