import React, {useState} from 'react';
import {View, StyleSheet} from 'react-native';
import {List, Switch, Divider, Text} from 'react-native-paper';
import {useDispatch, useSelector} from 'react-redux';
import {RootState} from '../../store';
import {setLanguage, setNotifications, setOfflineMode} from '../../store/slices/uiSlice';

const SettingsScreen: React.FC = () => {
  const dispatch = useDispatch();
  const ui = useSelector((s: RootState) => s.ui);
  const [notif, setNotif] = useState(ui.notifications);
  const [offline, setOffline] = useState(ui.offlineMode);

  return (
    <View style={styles.container}>
      <List.Section>
        <List.Subheader>Preferencias</List.Subheader>
        <List.Item
          title="Notificaciones"
          right={() => (
            <Switch
              value={notif}
              onValueChange={(v) => {
                setNotif(v);
                dispatch(setNotifications(v));
              }}
            />
          )}
        />
        <Divider />
        <List.Item
          title="Modo offline"
          right={() => (
            <Switch
              value={offline}
              onValueChange={(v) => {
                setOffline(v);
                dispatch(setOfflineMode(v));
              }}
            />
          )}
        />
        <Divider />
        <List.Item
          title="Idioma"
          description={ui.language.toUpperCase()}
          onPress={() =>
            dispatch(setLanguage(ui.language === 'es' ? 'en' : 'es'))
          }
        />
      </List.Section>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1},
});

export default SettingsScreen;
