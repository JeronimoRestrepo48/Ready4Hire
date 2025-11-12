import React, {useEffect} from 'react';
import {View, StyleSheet, FlatList} from 'react-native';
import {Card, Text, Chip} from 'react-native-paper';
import {useDispatch, useSelector} from 'react-redux';
import {fetchBadges, fetchUserBadges} from '../../store/slices/gamificationSlice';
import {RootState, AppDispatch} from '../../store';
import {getBadgeColor} from '../../utils';

const BadgesScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {badges, userBadges} = useSelector((s: RootState) => s.gamification);
  const {user} = useSelector((s: RootState) => s.auth);

  useEffect(() => {
    dispatch(fetchBadges() as any);
    if (user?.id) dispatch(fetchUserBadges(user.id) as any);
  }, [dispatch, user]);

  return (
    <View style={styles.container}>
      <FlatList
        data={badges}
        keyExtractor={(b: any) => String(b.id)}
        renderItem={({item}) => {
          const ub = userBadges.find((x: any) => x.badgeId === item.id);
          const unlocked = ub?.isUnlocked;
          const progress = Math.round((ub?.progress || 0) * 100);
          return (
            <Card style={styles.card}>
              <Card.Content>
                <Text variant="titleMedium">{item.icon} {item.name}</Text>
                <Text>{item.description}</Text>
                <View style={{flexDirection: 'row', marginTop: 8}}>
                  <Chip style={{backgroundColor: getBadgeColor(item.rarity), marginRight: 8}}>{item.rarity}</Chip>
                  <Chip>{unlocked ? 'Desbloqueado' : `${progress}%`}</Chip>
                </View>
              </Card.Content>
            </Card>
          );
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 12},
  card: {marginBottom: 8},
});

export default BadgesScreen;
