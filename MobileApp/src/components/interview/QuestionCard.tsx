/**
 * Question Card Component
 * Tarjeta para mostrar preguntas de entrevista
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Chip } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CustomCard } from '../common/Card';

interface QuestionCardProps {
  question: string;
  category?: string;
  difficulty?: 'junior' | 'mid' | 'senior';
  topic?: string;
  expectedConcepts?: string[];
  showDetails?: boolean;
  style?: any;
}

export const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  category,
  difficulty,
  topic,
  expectedConcepts,
  showDetails = false,
  style,
}) => {
  const getDifficultyColor = (level: string) => {
    switch (level) {
      case 'junior':
        return '#10b981';
      case 'mid':
        return '#f59e0b';
      case 'senior':
        return '#ef4444';
      default:
        return '#6366f1';
    }
  };

  const getDifficultyIcon = (level: string) => {
    switch (level) {
      case 'junior':
        return 'leaf';
      case 'mid':
        return 'fire';
      case 'senior':
        return 'lightning-bolt';
      default:
        return 'help-circle';
    }
  };

  return (
    <CustomCard variant="elevated" style={[styles.card, style]}>
      {/* Tags */}
      {(category || difficulty) && (
        <View style={styles.tags}>
          {category && (
            <Chip
              icon="tag"
              textStyle={styles.chipText}
              style={[styles.chip, styles.categoryChip]}
            >
              {category}
            </Chip>
          )}
          {difficulty && (
            <Chip
              icon={getDifficultyIcon(difficulty)}
              textStyle={[styles.chipText, { color: getDifficultyColor(difficulty) }]}
              style={[
                styles.chip,
                styles.difficultyChip,
                { backgroundColor: `${getDifficultyColor(difficulty)}20` },
              ]}
            >
              {difficulty}
            </Chip>
          )}
        </View>
      )}

      {/* Question Text */}
      <View style={styles.questionContainer}>
        <Icon name="help-circle" size={24} color="#6366f1" style={styles.questionIcon} />
        <Text style={styles.questionText}>{question}</Text>
      </View>

      {/* Topic */}
      {topic && (
        <View style={styles.topicContainer}>
          <Icon name="bookmark" size={16} color="#f59e0b" />
          <Text style={styles.topicText}>{topic}</Text>
        </View>
      )}

      {/* Expected Concepts (if showDetails) */}
      {showDetails && expectedConcepts && expectedConcepts.length > 0 && (
        <View style={styles.conceptsContainer}>
          <Text style={styles.conceptsTitle}>Conceptos esperados:</Text>
          <View style={styles.conceptsList}>
            {expectedConcepts.map((concept, index) => (
              <View key={index} style={styles.conceptItem}>
                <Icon name="check-circle-outline" size={16} color="#10b981" />
                <Text style={styles.conceptText}>{concept}</Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </CustomCard>
  );
};

const styles = StyleSheet.create({
  card: {
    marginBottom: 0,
  },
  tags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 12,
    gap: 8,
  },
  chip: {
    height: 32,
  },
  chipText: {
    fontSize: 12,
    fontWeight: '600',
  },
  categoryChip: {
    backgroundColor: '#6366f120',
  },
  difficultyChip: {
    borderWidth: 1,
  },
  questionContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  questionIcon: {
    marginRight: 12,
    marginTop: 2,
  },
  questionText: {
    flex: 1,
    fontSize: 18,
    lineHeight: 28,
    color: '#ffffff',
    fontWeight: '500',
  },
  topicContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#334155',
  },
  topicText: {
    fontSize: 14,
    color: '#fbbf24',
    marginLeft: 8,
    fontWeight: '500',
  },
  conceptsContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#334155',
  },
  conceptsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#94a3b8',
    marginBottom: 8,
  },
  conceptsList: {
    gap: 8,
  },
  conceptItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  conceptText: {
    fontSize: 14,
    color: '#e2e8f0',
    marginLeft: 8,
  },
});

export default QuestionCard;

