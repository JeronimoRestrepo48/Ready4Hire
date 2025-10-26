/**
 * Interview Screen
 * Pantalla principal de entrevistas
 */

import React, { useState } from 'react';
import { View, ScrollView, StyleSheet, Alert } from 'react-native';
import { Text, Card, Button, Title, SegmentedButtons, Chip } from 'react-native-paper';
import { Picker } from '@react-native-picker/picker';
import { useAuth } from '../../store/AuthContext';
import { useInterview } from '../../store/InterviewContext';

const ROLES = [
  'Software Engineer',
  'Frontend Developer',
  'Backend Developer',
  'Full Stack Developer',
  'Data Scientist',
  'DevOps Engineer',
  'QA Engineer',
  'Product Manager',
];

const InterviewScreen = ({ navigation }: any) => {
  const { user } = useAuth();
  const { startInterview, isLoading } = useInterview();
  
  const [selectedRole, setSelectedRole] = useState(ROLES[0]);
  const [mode, setMode] = useState('practice');
  const [skillLevel, setSkillLevel] = useState('junior');

  const handleStartInterview = async () => {
    if (!user) {
      Alert.alert('Error', 'Debes iniciar sesión primero');
      return;
    }

    try {
      await startInterview(user.id, selectedRole, mode, skillLevel);
      navigation.navigate('InterviewActive');
    } catch (error: any) {
      Alert.alert('Error', error.message || 'No se pudo iniciar la entrevista');
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title style={styles.title}>Nueva Entrevista</Title>
        <Text style={styles.subtitle}>Configura tu sesión de práctica</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Modo de Entrevista</Text>
          <SegmentedButtons
            value={mode}
            onValueChange={setMode}
            buttons={[
              { 
                value: 'practice', 
                label: '🎓 Práctica',
                style: mode === 'practice' ? styles.activeButton : {}
              },
              { 
                value: 'exam', 
                label: '📝 Examen',
                style: mode === 'exam' ? styles.activeButton : {}
              },
            ]}
            style={styles.segmented}
          />
          <View style={styles.modeInfo}>
            {mode === 'practice' ? (
              <>
                <Chip icon="lightbulb" style={styles.chip}>3 intentos</Chip>
                <Chip icon="help-circle" style={styles.chip}>Pistas disponibles</Chip>
                <Chip icon="clock-outline" style={styles.chip}>Sin límite</Chip>
              </>
            ) : (
              <>
                <Chip icon="alert" style={styles.chip}>1 intento</Chip>
                <Chip icon="cancel" style={styles.chip}>Sin pistas</Chip>
                <Chip icon="timer" style={styles.chip}>Tiempo límite</Chip>
              </>
            )}
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Posición</Text>
          <View style={styles.pickerContainer}>
            <Picker
              selectedValue={selectedRole}
              onValueChange={setSelectedRole}
              style={styles.picker}
            >
              {ROLES.map(role => (
                <Picker.Item key={role} label={role} value={role} />
              ))}
            </Picker>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Text style={styles.sectionTitle}>Nivel de Experiencia</Text>
          <SegmentedButtons
            value={skillLevel}
            onValueChange={setSkillLevel}
            buttons={[
              { value: 'junior', label: 'Junior' },
              { value: 'mid', label: 'Mid' },
              { value: 'senior', label: 'Senior' },
            ]}
            style={styles.segmented}
          />
        </Card.Content>
      </Card>

      <Button
        mode="contained"
        onPress={handleStartInterview}
        loading={isLoading}
        disabled={isLoading}
        style={styles.startButton}
        buttonColor="#6366f1"
        icon="play-circle"
      >
        Iniciar Entrevista
      </Button>
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
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  segmented: {
    marginVertical: 8,
  },
  activeButton: {
    backgroundColor: '#6366f1',
  },
  modeInfo: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 12,
    gap: 8,
  },
  chip: {
    marginRight: 8,
    marginBottom: 8,
  },
  pickerContainer: {
    backgroundColor: '#0f172a',
    borderRadius: 8,
    overflow: 'hidden',
  },
  picker: {
    color: '#ffffff',
  },
  startButton: {
    margin: 16,
    paddingVertical: 8,
  },
});

export default InterviewScreen;

