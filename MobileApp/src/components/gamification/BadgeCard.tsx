/**
 * Badge Card Component
 * Tarjeta para mostrar badges/logros
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CustomCard } from '../common/Card';

interface BadgeCardProps {
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  earned?: boolean;
  earnedDate?: string;
  progress?: number;
  xpReward?: number;
  onPress?: () => void;
  style?: any;
}

export const BadgeCard: React.FC<BadgeCardProps> = ({
  name,
  description,
  icon,
  rarity,
  earned = false,
  earnedDate,
  progress,
  xpReward,
  onPress,
  style,
}) => {
  const getRarityColor = (level: string) => {
    switch (level) {
      case 'common':
        return '#94a3b8';
      case 'rare':
        return '#3b82f6';
      case 'epic':
        return '#a855f7';
      case 'legendary':
        return '#f59e0b';
      default:
        return '#6366f1';
    }
  };

  const getRarityGradient = (level: string) => {
    switch (level) {
      case 'common':
        return ['#94a3b8', '#64748b'];
      case 'rare':
        return ['#3b82f6', '#2563eb'];
      case 'epic':
        return ['#a855f7', '#9333ea'];
      case 'legendary':
        return ['#f59e0b', '#d97706'];
      default:
        return ['#6366f1', '#4f46e5'];
    }
  };

  return (
    <CustomCard
      variant="elevated"
      onPress={onPress}
      style={[
        styles.card,
        !earned && styles.cardLocked,
        { borderColor: getRarityColor(rarity), borderWidth: 2 },
        style,
      ]}
    >
      <View style={styles.container}>
        {/* Badge Icon */}
        <View
          style={[
            styles.iconContainer,
            { backgroundColor: `${getRarityColor(rarity)}30` },
            !earned && styles.iconContainerLocked,
          ]}
        >
          <Icon
            name={earned ? icon : 'lock'}
            size={48}
            color={earned ? getRarityColor(rarity) : '#64748b'}
          />
          {earned && (
            <View style={[styles.rarityIndicator, { backgroundColor: getRarityColor(rarity) }]} />
          )}
        </View>

        {/* Badge Info */}
        <View style={styles.info}>
          <Text style={[styles.name, !earned && styles.nameLocked]}>
            {earned ? name : '???'}
          </Text>
          
          <View style={styles.rarityBadge}>
            <Icon name="star" size={12} color={getRarityColor(rarity)} />
            <Text style={[styles.rarityText, { color: getRarityColor(rarity) }]}>
              {rarity.toUpperCase()}
            </Text>
          </View>

          <Text style={[styles.description, !earned && styles.descriptionLocked]}>
            {earned ? description : 'Completa el desafío para desbloquear'}
          </Text>

          {/* Progress Bar (if not earned and progress available) */}
          {!earned && progress !== undefined && (
            <View style={styles.progressContainer}>
              <View style={styles.progressBar}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${progress}%`,
                      backgroundColor: getRarityColor(rarity),
                    },
                  ]}
                />
              </View>
              <Text style={styles.progressText}>{Math.round(progress)}%</Text>
            </View>
          )}

          {/* Earned Date */}
          {earned && earnedDate && (
            <View style={styles.earnedContainer}>
              <Icon name="calendar-check" size={14} color="#10b981" />
              <Text style={styles.earnedText}>Obtenido: {earnedDate}</Text>
            </View>
          )}

          {/* XP Reward */}
          {xpReward && (
            <View style={styles.rewardContainer}>
              <Icon name="star" size={14} color="#f59e0b" />
              <Text style={styles.rewardText}>+{xpReward} XP</Text>
            </View>
          )}
        </View>
      </View>
    </CustomCard>
  );
};

const styles = StyleSheet.create({
  card: {
    marginBottom: 12,
  },
  cardLocked: {
    opacity: 0.7,
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
    position: 'relative',
  },
  iconContainerLocked: {
    backgroundColor: '#1e293b',
  },
  rarityIndicator: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#1e293b',
  },
  info: {
    flex: 1,
  },
  name: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  nameLocked: {
    color: '#64748b',
  },
  rarityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  rarityText: {
    fontSize: 10,
    fontWeight: 'bold',
    marginLeft: 4,
    letterSpacing: 1,
  },
  description: {
    fontSize: 14,
    color: '#94a3b8',
    lineHeight: 20,
  },
  descriptionLocked: {
    fontStyle: 'italic',
  },
  progressContainer: {
    marginTop: 12,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#334155',
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 4,
  },
  progressFill: {
    height: '100%',
  },
  progressText: {
    fontSize: 12,
    color: '#94a3b8',
    textAlign: 'right',
  },
  earnedContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  earnedText: {
    fontSize: 12,
    color: '#10b981',
    marginLeft: 4,
  },
  rewardContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  rewardText: {
    fontSize: 12,
    color: '#f59e0b',
    marginLeft: 4,
    fontWeight: 'bold',
  },
});

export default BadgeCard;

