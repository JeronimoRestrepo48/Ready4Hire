import React from 'react';
import {View, StyleSheet} from 'react-native';
import {Text, Card, Button} from 'react-native-paper';

const InterviewListScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Card>
        <Card.Content>
          <Text>No hay entrevistas previas.</Text>
          <Button style={{marginTop: 8}} mode="contained">
            Iniciar nueva entrevista
          </Button>
        </Card.Content>
      </Card>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, padding: 16},
});

export default InterviewListScreen;
