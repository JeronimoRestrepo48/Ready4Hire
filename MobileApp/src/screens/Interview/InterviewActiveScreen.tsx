/**
 * Interview Active Screen
 * Pantalla de entrevista activa con preguntas en vivo
 */

import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { Text, Card, Button, TextInput, ProgressBar, Chip, Portal, Modal } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useAuth } from '../../store/AuthContext';
import { useInterview } from '../../store/InterviewContext';

const InterviewActiveScreen = ({ route, navigation }: any) => {
  const { user } = useAuth();
  const { activeInterview, processAnswer, completeInterview, isLoading } = useInterview();
  
  const [answer, setAnswer] = useState('');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answeredQuestions, setAnsweredQuestions] = useState<string[]>([]);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const [hintLevel, setHintLevel] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [feedback, setFeedback] = useState<any>(null);

  const currentQuestion = activeInterview?.currentQuestion;
  const totalQuestions = activeInterview?.totalQuestions || 10;
  const progress = answeredQuestions.length / totalQuestions;
  const isExamMode = activeInterview?.mode === 'exam';
  const maxAttempts = isExamMode ? 1 : 3;

  // Timer
  useEffect(() => {
    const timer = setInterval(() => {
      setTimeElapsed(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSubmitAnswer = async () => {
    if (!answer.trim()) {
      Alert.alert('Error', 'Por favor escribe una respuesta');
      return;
    }

    if (!activeInterview?.id) {
      Alert.alert('Error', 'No hay entrevista activa');
      return;
    }

    try {
      const result = await processAnswer(
        activeInterview.id,
        answer,
        timeElapsed
      );

      if (result) {
        setFeedback(result);
        setAttempts(prev => prev + 1);

        if (result.isCorrect || attempts >= maxAttempts - 1) {
          // Pregunta completada
          setAnsweredQuestions(prev => [...prev, currentQuestion?.id || '']);
          
          // Esperar 3 segundos para ver feedback
          setTimeout(() => {
            setAnswer('');
            setFeedback(null);
            setAttempts(0);
            setHintLevel(0);
            setShowHint(false);
            setCurrentQuestionIndex(prev => prev + 1);
          }, 3000);
        }
      }
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Error al procesar respuesta');
    }
  };

  const handleRequestHint = () => {
    if (isExamMode) {
      Alert.alert('No disponible', 'Las pistas no están disponibles en modo examen');
      return;
    }

    if (hintLevel < 3) {
      setHintLevel(prev => prev + 1);
      setShowHint(true);
    }
  };

  const handleEndInterview = () => {
    Alert.alert(
      'Finalizar Entrevista',
      '¿Estás seguro que deseas finalizar? Se generará tu reporte.',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Finalizar',
          style: 'destructive',
          onPress: async () => {
            if (activeInterview?.id) {
              const report = await completeInterview(activeInterview.id);
              navigation.navigate('InterviewResult', { report });
            }
          },
        },
      ]
    );
  };

  if (!activeInterview || !currentQuestion) {
    return (
      <View style={styles.centerContainer}>
        <Icon name="alert-circle" size={64} color="#ef4444" />
        <Text style={styles.errorText}>No hay entrevista activa</Text>
        <Button mode="contained" onPress={() => navigation.goBack()}>
          Volver
        </Button>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerInfo}>
            <Text style={styles.roleText}>{activeInterview.role}</Text>
            <View style={styles.modeChip}>
              <Chip
                icon={isExamMode ? 'certificate' : 'school'}
                style={[styles.chip, isExamMode ? styles.examChip : styles.practiceChip]}
              >
                {isExamMode ? 'Examen' : 'Práctica'}
              </Chip>
              <Chip icon="clock-outline" style={styles.chip}>
                {formatTime(timeElapsed)}
              </Chip>
            </View>
          </View>
          <Button mode="text" onPress={handleEndInterview} textColor="#ef4444">
            Finalizar
          </Button>
        </View>

        {/* Progress */}
        <Card style={styles.progressCard}>
          <Card.Content>
            <View style={styles.progressHeader}>
              <Text style={styles.progressText}>
                Pregunta {answeredQuestions.length + 1} de {totalQuestions}
              </Text>
              <Text style={styles.progressPercentage}>
                {Math.round(progress * 100)}%
              </Text>
            </View>
            <ProgressBar progress={progress} color="#6366f1" style={styles.progressBar} />
          </Card.Content>
        </Card>

        {/* Question Card */}
        <Card style={styles.questionCard}>
          <Card.Content>
            <View style={styles.questionHeader}>
              <Chip icon="head-question" style={styles.categoryChip}>
                {currentQuestion.category}
              </Chip>
              <Chip
                icon={
                  currentQuestion.difficulty === 'junior'
                    ? 'leaf'
                    : currentQuestion.difficulty === 'mid'
                    ? 'fire'
                    : 'lightning-bolt'
                }
                style={styles.difficultyChip}
              >
                {currentQuestion.difficulty}
              </Chip>
            </View>
            <Text style={styles.questionText}>{currentQuestion.text}</Text>
            
            {currentQuestion.topic && (
              <Chip icon="tag" style={styles.topicChip}>
                {currentQuestion.topic}
              </Chip>
            )}
          </Card.Content>
        </Card>

        {/* Hint Card */}
        {showHint && (
          <Card style={styles.hintCard}>
            <Card.Content>
              <View style={styles.hintHeader}>
                <Icon name="lightbulb" size={24} color="#f59e0b" />
                <Text style={styles.hintTitle}>Pista {hintLevel}</Text>
              </View>
              <Text style={styles.hintText}>
                {hintLevel === 1
                  ? 'Piensa en los conceptos clave relacionados con el tema.'
                  : hintLevel === 2
                  ? 'Considera las mejores prácticas y patrones comunes.'
                  : 'Recuerda incluir ejemplos específicos en tu respuesta.'}
              </Text>
            </Card.Content>
          </Card>
        )}

        {/* Feedback Card */}
        {feedback && (
          <Card
            style={[
              styles.feedbackCard,
              feedback.isCorrect ? styles.correctFeedback : styles.incorrectFeedback,
            ]}
          >
            <Card.Content>
              <View style={styles.feedbackHeader}>
                <Icon
                  name={feedback.isCorrect ? 'check-circle' : 'close-circle'}
                  size={32}
                  color={feedback.isCorrect ? '#10b981' : '#ef4444'}
                />
                <Text style={styles.feedbackTitle}>
                  {feedback.isCorrect ? '¡Correcto!' : 'Intenta de nuevo'}
                </Text>
              </View>
              <Text style={styles.scoreText}>Puntuación: {feedback.score}/10</Text>
              <Text style={styles.feedbackText}>{feedback.feedback}</Text>
            </Card.Content>
          </Card>
        )}

        {/* Answer Input */}
        <Card style={styles.answerCard}>
          <Card.Content>
            <Text style={styles.answerLabel}>Tu respuesta:</Text>
            <TextInput
              value={answer}
              onChangeText={setAnswer}
              mode="outlined"
              multiline
              numberOfLines={6}
              placeholder="Escribe tu respuesta aquí..."
              style={styles.answerInput}
              outlineColor="#334155"
              activeOutlineColor="#6366f1"
              theme={{ colors: { background: '#1e293b' } }}
              disabled={isLoading || !!feedback}
            />
            
            <View style={styles.answerActions}>
              {!isExamMode && attempts < maxAttempts && !feedback && (
                <Button
                  mode="text"
                  icon="lightbulb-outline"
                  onPress={handleRequestHint}
                  disabled={hintLevel >= 3}
                >
                  Pista ({hintLevel}/3)
                </Button>
              )}
              
              <View style={styles.submitButtonContainer}>
                {!isExamMode && (
                  <Text style={styles.attemptsText}>
                    Intento {attempts + 1}/{maxAttempts}
                  </Text>
                )}
                <Button
                  mode="contained"
                  onPress={handleSubmitAnswer}
                  loading={isLoading}
                  disabled={isLoading || !answer.trim() || !!feedback}
                  buttonColor="#6366f1"
                >
                  Enviar Respuesta
                </Button>
              </View>
            </View>
          </Card.Content>
        </Card>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0e1a',
  },
  scrollView: {
    flex: 1,
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 48,
  },
  headerInfo: {
    flex: 1,
  },
  roleText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  modeChip: {
    flexDirection: 'row',
    gap: 8,
  },
  chip: {
    marginRight: 8,
  },
  examChip: {
    backgroundColor: '#ef444420',
  },
  practiceChip: {
    backgroundColor: '#10b98120',
  },
  progressCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#1e293b',
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  progressText: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: 'bold',
  },
  progressPercentage: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: 'bold',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  questionCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#1e293b',
  },
  questionHeader: {
    flexDirection: 'row',
    marginBottom: 16,
    gap: 8,
  },
  categoryChip: {
    backgroundColor: '#6366f120',
  },
  difficultyChip: {
    backgroundColor: '#f59e0b20',
  },
  topicChip: {
    marginTop: 12,
    alignSelf: 'flex-start',
  },
  questionText: {
    fontSize: 18,
    color: '#ffffff',
    lineHeight: 28,
  },
  hintCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#f59e0b20',
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  hintHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  hintTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#f59e0b',
    marginLeft: 8,
  },
  hintText: {
    fontSize: 14,
    color: '#fbbf24',
    lineHeight: 20,
  },
  feedbackCard: {
    margin: 16,
    marginTop: 0,
  },
  correctFeedback: {
    backgroundColor: '#10b98120',
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
  },
  incorrectFeedback: {
    backgroundColor: '#ef444420',
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  feedbackHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  feedbackTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginLeft: 12,
  },
  scoreText: {
    fontSize: 16,
    color: '#94a3b8',
    marginBottom: 8,
  },
  feedbackText: {
    fontSize: 14,
    color: '#e2e8f0',
    lineHeight: 20,
  },
  answerCard: {
    margin: 16,
    marginTop: 0,
    backgroundColor: '#1e293b',
  },
  answerLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  answerInput: {
    marginBottom: 16,
    fontSize: 14,
    color: '#ffffff',
  },
  answerActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  submitButtonContainer: {
    alignItems: 'flex-end',
  },
  attemptsText: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 8,
  },
});

export default InterviewActiveScreen;

