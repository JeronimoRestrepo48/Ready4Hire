/**
 * Interview Screen (Chat UI)
 */

import React, {useEffect, useState} from 'react';
import {View, StyleSheet, FlatList} from 'react-native';
import {Appbar, TextInput, Button, Card, Text} from 'react-native-paper';
import {useDispatch, useSelector} from 'react-redux';
import {startInterview, processAnswer} from '../../store/slices/interviewSlice';
import {RootState, AppDispatch} from '../../store';
import {theme} from '../../theme';

const InterviewScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {user} = useSelector((s: RootState) => s.auth);
  const interviewState = useSelector((s: RootState) => s.interview);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<{role: 'system' | 'user' | 'ai'; text: string}[]>([]);

  useEffect(() => {
    // If no interview, start one with defaults
    if (!interviewState.currentInterview && user?.id) {
      dispatch(
        startInterview({
          userId: user.id,
          role: user.profession || 'Backend Developer',
          difficulty: 'mid',
          category: 'technical',
        }) as any
      ).then((res: any) => {
        const first = res.payload?.firstQuestion?.text;
        if (first) {
          setMessages((prev) => [...prev, {role: 'ai', text: first}]);
        }
      });
    }
  }, [user, interviewState.currentInterview, dispatch]);

  const onSend = async () => {
    if (!input.trim() || !interviewState.currentInterview) return;

    const {id} = interviewState.currentInterview;
    const text = input.trim();
    setInput('');
    setMessages((prev) => [...prev, {role: 'user', text}]);

    const res: any = await dispatch(
      processAnswer({interviewId: id, answer: text, timeTaken: 30}) as any
    );

    const feedback = res.payload?.feedback as string | undefined;
    const nextQ = res.payload?.nextQuestion?.text as string | undefined;

    if (feedback) setMessages((prev) => [...prev, {role: 'system', text: feedback}]);
    if (nextQ) setMessages((prev) => [...prev, {role: 'ai', text: nextQ}]);
  };

  return (
    <View style={styles.container}>
      <Appbar.Header>
        <Appbar.Content title="Entrevista" />
      </Appbar.Header>

      <FlatList
        data={messages}
        keyExtractor={(_, idx) => String(idx)}
        contentContainerStyle={styles.list}
        renderItem={({item}) => (
          <Card style={[styles.msg, item.role === 'user' ? styles.user : styles.ai]}>
            <Card.Content>
              <Text style={styles.msgText}>{item.text}</Text>
            </Card.Content>
          </Card>
        )}
      />

      <View style={styles.inputBar}>
        <TextInput
          style={{flex: 1, marginRight: 8}}
          mode="outlined"
          placeholder="Escribe tu respuesta..."
          value={input}
          onChangeText={setInput}
        />
        <Button mode="contained" onPress={onSend} disabled={interviewState.loading}>
          Enviar
        </Button>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: theme.colors.background},
  list: {padding: 12},
  msg: {marginBottom: 8},
  user: {alignSelf: 'flex-end', backgroundColor: '#E5E7EB', maxWidth: '80%'},
  ai: {alignSelf: 'flex-start', backgroundColor: '#FFF', maxWidth: '80%'},
  msgText: {fontSize: 15, color: theme.colors.text},
  inputBar: {
    flexDirection: 'row',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    backgroundColor: '#fff',
  },
});

export default InterviewScreen;
