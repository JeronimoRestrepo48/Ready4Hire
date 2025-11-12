/**
 * Navigation Type Definitions
 */

import {StackScreenProps} from '@react-navigation/stack';
import {BottomTabScreenProps} from '@react-navigation/bottom-tabs';
import {CompositeScreenProps} from '@react-navigation/native';
import {RootStackParamList} from './index';

export type AuthScreenProps<T extends keyof RootStackParamList = 'Login'> = StackScreenProps<
  RootStackParamList,
  T
>;

export type HomeScreenProps = BottomTabScreenProps<RootStackParamList, 'Home'>;
export type InterviewScreenProps = StackScreenProps<RootStackParamList, 'Interview'>;
export type ProfileScreenProps = BottomTabScreenProps<RootStackParamList, 'Profile'>;

