import React, {useEffect} from 'react';
import {View, StyleSheet, FlatList} from 'react-native';
import {Card, Button, Text} from 'react-native-paper';
import {useDispatch, useSelector} from 'react-redux';
import {fetchGames} from '../../store/slices/gamificationSlice';
import {RootState, AppDispatch} from '../../store';
import {useNavigation} from '@react-navigation/native';

const GamesScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {games} = useSelector((s: RootState) => s.gamification);
  const nav = useNavigation<any>();

  useEffect(() => {
    dispatch(fetchGames() as any);
  }, [dispatch]);

  return (
    <View style={styles.container}>
      <FlatList
        data={games}
        keyExtractor={(g: any) => g.id}
        renderItem={({item}) => (
          <Card style={styles.card}>
            <Card.Title title={item.name} subtitle={`${item.pointsReward} pts`} />
            <Card.Content>
              <Text>{item.description}</Text>
            </Card.Content>
            <Card.Actions>
              <Button onPress={() => nav.navigate('GameSession', {gameId: item.id})}>Jugar</Button>
            </Card.Actions>
          </Card>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 12},
  card: {marginBottom: 8},
});

export default GamesScreen;
