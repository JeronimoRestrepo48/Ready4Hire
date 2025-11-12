import React from 'react';
import {View, StyleSheet} from 'react-native';
import {Text, Button} from 'react-native-paper';

const GameSessionScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text variant="titleMedium">Sesión de juego</Text>
      <Text style={{marginTop: 8}}>Contenido del juego generado por IA aparecerá aquí.</Text>
      <Button style={{marginTop: 16}} mode="contained">Enviar respuesta</Button>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 16},
});

export default GameSessionScreen;
