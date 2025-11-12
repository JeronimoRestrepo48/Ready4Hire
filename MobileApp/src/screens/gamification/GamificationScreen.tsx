import React from 'react';
import {View, StyleSheet} from 'react-native';
import {Card, Button, Text} from 'react-native-paper';
import {useNavigation} from '@react-navigation/native';

const GamificationScreen: React.FC = () => {
  const nav = useNavigation<any>();

  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Card.Title title="Badges" subtitle="Tus logros y progreso" />
        <Card.Actions>
          <Button onPress={() => nav.navigate('Badges')}>Ver Badges</Button>
        </Card.Actions>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Juegos" subtitle="Practica con IA" />
        <Card.Actions>
          <Button onPress={() => nav.navigate('Games')}>Ver Juegos</Button>
        </Card.Actions>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Ranking" subtitle="Leaderboard global" />
        <Card.Actions>
          <Button onPress={() => nav.navigate('Leaderboard')}>Ver Ranking</Button>
        </Card.Actions>
      </Card>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 16},
  card: {marginBottom: 12},
});

export default GamificationScreen;
