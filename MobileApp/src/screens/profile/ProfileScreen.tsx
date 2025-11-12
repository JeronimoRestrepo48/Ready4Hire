import React from 'react';
import {View, StyleSheet} from 'react-native';
import {Avatar, Card, Text} from 'react-native-paper';
import {useSelector} from 'react-redux';
import {RootState} from '../../store';

const ProfileScreen: React.FC = () => {
  const {user} = useSelector((s: RootState) => s.auth);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Avatar.Icon size={72} icon="account" />
        <Text style={styles.name}>{user?.name} {user?.lastName}</Text>
        <Text>{user?.email}</Text>
      </View>

      <Card style={styles.card}>
        <Card.Title title="Profesión" />
        <Card.Content>
          <Text>{user?.profession || 'No especificada'}</Text>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="País" />
        <Card.Content>
          <Text>{user?.country || 'No especificado'}</Text>
        </Card.Content>
      </Card>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 16},
  header: {alignItems: 'center', marginBottom: 16},
  name: {fontSize: 20, fontWeight: '600', marginTop: 8},
  card: {marginBottom: 8},
});

export default ProfileScreen;
