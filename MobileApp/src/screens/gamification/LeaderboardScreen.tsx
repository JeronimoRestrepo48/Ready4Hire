import React, {useEffect} from 'react';
import {View, StyleSheet, FlatList} from 'react-native';
import {List, Text} from 'react-native-paper';
import {useDispatch, useSelector} from 'react-redux';
import {fetchLeaderboard} from '../../store/slices/gamificationSlice';
import {RootState, AppDispatch} from '../../store';

const LeaderboardScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {leaderboard} = useSelector((s: RootState) => s.gamification);

  useEffect(() => {
    dispatch(fetchLeaderboard() as any);
  }, [dispatch]);

  return (
    <View style={styles.container}>
      <FlatList
        data={leaderboard}
        keyExtractor={(x: any) => String(x.rank)}
        renderItem={({item}) => (
          <List.Item
            title={`#${item.rank} ${item.username}`}
            description={`${item.totalPoints} pts • Nivel ${item.level}`}
            left={() => <List.Icon icon="trophy" />}
          />
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1},
});

export default LeaderboardScreen;
