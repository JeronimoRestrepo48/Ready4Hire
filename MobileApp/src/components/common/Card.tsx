/**
 * Custom Card Component
 * Tarjeta reutilizable con variantes y efectos
 */

import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { Text } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

interface CustomCardProps {
  children?: React.ReactNode;
  title?: string;
  subtitle?: string;
  icon?: string;
  iconColor?: string;
  variant?: 'default' | 'outlined' | 'elevated';
  onPress?: () => void;
  style?: any;
}

export const CustomCard: React.FC<CustomCardProps> = ({
  children,
  title,
  subtitle,
  icon,
  iconColor = '#6366f1',
  variant = 'default',
  onPress,
  style,
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'outlined':
        return styles.outlinedCard;
      case 'elevated':
        return styles.elevatedCard;
      default:
        return styles.defaultCard;
    }
  };

  const CardContent = (
    <View style={[styles.card, getVariantStyles(), style]}>
      {(title || icon) && (
        <View style={styles.header}>
          {icon && (
            <View style={[styles.iconContainer, { backgroundColor: `${iconColor}20` }]}>
              <Icon name={icon} size={24} color={iconColor} />
            </View>
          )}
          <View style={styles.headerText}>
            {title && <Text style={styles.title}>{title}</Text>}
            {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
          </View>
        </View>
      )}
      {children && <View style={styles.content}>{children}</View>}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {CardContent}
      </TouchableOpacity>
    );
  }

  return CardContent;
};

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  defaultCard: {
    backgroundColor: '#1e293b',
  },
  outlinedCard: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#334155',
  },
  elevatedCard: {
    backgroundColor: '#1e293b',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  headerText: {
    flex: 1,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  subtitle: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  content: {
    marginTop: 8,
  },
});

export default CustomCard;

