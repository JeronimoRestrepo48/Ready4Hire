/**
 * Profile Screen
 * Pantalla de perfil de usuario
 */

import React from 'react';
import { View, ScrollView, StyleSheet, Alert } from 'react-native';
import { Text, Card, Button, Title, Avatar, List, Divider } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useAuth } from '../../store/AuthContext';

const ProfileScreen = ({ navigation }: any) => {
  const { user, logout } = useAuth();

  const handleLogout = () => {
    Alert.alert(
      'Cerrar Sesión',
      '¿Estás seguro que deseas cerrar sesión?',
      [
        { text: 'Cancelar', style: 'cancel' },
        { 
          text: 'Cerrar Sesión', 
          style: 'destructive',
          onPress: async () => {
            await logout();
          }
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Avatar.Text 
          size={80} 
          label={user?.name?.charAt(0) || 'U'} 
          style={styles.avatar}
        />
        <Title style={styles.name}>
          {user?.name} {user?.lastName}
        </Title>
        <Text style={styles.email}>{user?.email}</Text>
      </View>

      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Icon name="trophy" size={32} color="#f59e0b" />
              <Text style={styles.statValue}>{user?.level || 1}</Text>
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
          <List.Section>
            <List.Subheader style={styles.subheader}>Configuración</List.Subheader>
            
            <List.Item
              title="Editar Perfil"
              description="Actualiza tu información personal"
              left={props => <List.Icon {...props} icon="account-edit" color="#6366f1" />}
              right={props => <List.Icon {...props} icon="chevron-right" />}
              onPress={() => console.log('Edit profile')}
              titleStyle={styles.listTitle}
              descriptionStyle={styles.listDescription}
            />
            <Divider />
            
            <List.Item
              title="Habilidades"
              description="Gestiona tus habilidades técnicas"
              left={props => <List.Icon {...props} icon="brain" color="#10b981" />}
              right={props => <List.Icon {...props} icon="chevron-right" />}
              onPress={() => console.log('Edit skills')}
              titleStyle={styles.listTitle}
              descriptionStyle={styles.listDescription}
            />
            <Divider />
            
            <List.Item
              title="Notificaciones"
              description="Configura tus notificaciones"
              left={props => <List.Icon {...props} icon="bell" color="#f59e0b" />}
              right={props => <List.Icon {...props} icon="chevron-right" />}
              onPress={() => console.log('Notifications')}
              titleStyle={styles.listTitle}
              descriptionStyle={styles.listDescription}
            />
            <Divider />
            
            <List.Item
              title="Privacidad"
              description="Gestiona tu privacidad"
              left={props => <List.Icon {...props} icon="shield-check" color="#6366f1" />}
              right={props => <List.Icon {...props} icon="chevron-right" />}
              onPress={() => console.log('Privacy')}
              titleStyle={styles.listTitle}
              descriptionStyle={styles.listDescription}
            />
            <Divider />
            
            <List.Item
              title="Ayuda y Soporte"
              description="Obtén ayuda"
              left={props => <List.Icon {...props} icon="help-circle" color="#94a3b8" />}
              right={props => <List.Icon {...props} icon="chevron-right" />}
              onPress={() => console.log('Help')}
              titleStyle={styles.listTitle}
              descriptionStyle={styles.listDescription}
            />
          </List.Section>
        </Card.Content>
      </Card>

      <Button
        mode="contained"
        onPress={handleLogout}
        icon="logout"
        style={styles.logoutButton}
        buttonColor="#ef4444"
      >
        Cerrar Sesión
      </Button>

      <Text style={styles.version}>Ready4Hire v1.0.0</Text>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0e1a',
  },
  header: {
    alignItems: 'center',
    padding: 24,
    paddingTop: 48,
  },
  avatar: {
    backgroundColor: '#6366f1',
    marginBottom: 16,
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  email: {
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
  subheader: {
    color: '#94a3b8',
    fontSize: 14,
    fontWeight: 'bold',
  },
  listTitle: {
    color: '#ffffff',
    fontSize: 16,
  },
  listDescription: {
    color: '#94a3b8',
    fontSize: 14,
  },
  logoutButton: {
    margin: 16,
    paddingVertical: 8,
  },
  version: {
    textAlign: 'center',
    color: '#64748b',
    fontSize: 12,
    paddingBottom: 24,
  },
});

export default ProfileScreen;

