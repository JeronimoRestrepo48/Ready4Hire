/**
 * TypeScript Definitions for Ready4Hire Mobile App
 */

// User Types
export interface User {
  id: string;
  email: string;
  name: string;
  lastName?: string;
  profession?: string;
  professionCategory?: string;
  experienceLevel?: 'junior' | 'mid' | 'senior';
  skills?: string[];
  softSkills?: string[];
  interests?: string[];
  country?: string;
  avatarUrl?: string;
  bio?: string;
}

// Authentication Types
export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
  lastName?: string;
}

// Interview Types
export interface Interview {
  id: string;
  userId: string;
  role: string;
  skillLevel: 'junior' | 'mid' | 'senior';
  interviewType: 'technical' | 'soft_skills';
  currentPhase: 'context' | 'questions';
  status: 'created' | 'active' | 'completed' | 'paused';
  questions: string[];
  answers: Answer[];
  contextAnswers: string[];
  currentQuestionIndex: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

export interface Question {
  id: string;
  text: string;
  category: string;
  difficulty: 'junior' | 'mid' | 'senior';
  topic: string;
  expectedConcepts?: string[];
  hintsAvailable?: boolean;
}

export interface Answer {
  questionId: string;
  text: string;
  score: number;
  isCorrect: boolean;
  emotion?: Emotion;
  timeTaken?: number;
  evaluationDetails?: any;
  feedback?: string;
  createdAt: string;
}

export interface Emotion {
  emotion: 'joy' | 'sadness' | 'anger' | 'fear' | 'surprise' | 'neutral';
  confidence: number;
  language?: string;
}

export interface Evaluation {
  score: number;
  isCorrect: boolean;
  feedback: string;
  breakdown?: {
    completeness: number;
    technicalDepth: number;
    clarity: number;
    keyConcepts: number;
  };
  strengths?: string[];
  improvements?: string[];
  conceptsCovered?: string[];
  missingConcepts?: string[];
}

export interface InterviewProgress {
  contextCompleted: number;
  questionsCompleted: number;
  totalQuestions: number;
  percentage?: number;
}

export interface InterviewResponse {
  interviewId: string;
  firstQuestion?: Question;
  nextQuestion?: Question;
  status: string;
  message: string;
  phase: 'context' | 'questions' | 'completed';
  progress?: InterviewProgress;
  evaluation?: Evaluation;
  feedback?: string;
  emotion?: Emotion;
  interviewCompleted?: boolean;
}

// API Requests
export interface StartInterviewRequest {
  userId: string;
  role: string;
  difficulty: 'junior' | 'mid' | 'senior';
  category: 'technical' | 'soft_skills';
}

export interface ProcessAnswerRequest {
  interviewId: string;
  answer: string;
  timeTaken?: number;
}

// Gamification Types
export interface UserStats {
  userId: string;
  level: number;
  experience: number;
  totalPoints: number;
  totalGamesPlayed: number;
  totalGamesWon: number;
  streakDays: number;
  rank: number;
  gamesByType?: Record<string, number>;
  bestScores?: Record<string, number>;
}

export interface Badge {
  id: number;
  name: string;
  description: string;
  icon: string;
  category: 'general' | 'technical' | 'soft_skills' | 'achievement' | 'milestone';
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  pointsRequired: number;
  requirementType: string;
  requirementValue: number;
  rewardPoints: number;
  rewardXp: number;
  isActive: boolean;
}

export interface UserBadge {
  badgeId: number;
  badgeName: string;
  progress: number; // 0.0 to 1.0
  isUnlocked: boolean;
  unlockedAt?: string;
}

export interface Game {
  id: string;
  name: string;
  description: string;
  icon: string;
  type: string;
  profession: string;
  difficulty: 'easy' | 'medium' | 'hard';
  durationMinutes: number;
  pointsReward: number;
  aiPowered: boolean;
}

export interface GameSession {
  sessionId: string;
  gameId: string;
  gameType: string;
  challenge: any;
  rules: string[];
  timeLimitMinutes: number;
}

export interface LeaderboardEntry {
  rank: number;
  userId: string;
  username: string;
  totalPoints: number;
  level: number;
  gamesWon: number;
  achievementsCount: number;
}

// API Types
export interface APIError {
  error: string;
  message: string;
  timestamp?: string;
}

export interface APIResponse<T> {
  data?: T;
  error?: APIError;
}

// Navigation Types
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  Login: undefined;
  Register: undefined;
  Home: undefined;
  Interview: {interviewId?: string};
  InterviewList: undefined;
  Gamification: undefined;
  Profile: undefined;
  Settings: undefined;
  Leaderboard: undefined;
  Games: undefined;
  Badges: undefined;
  GameSession: {gameId: string};
};

// Redux Types
export interface RootState {
  auth: AuthState;
  interview: InterviewState;
  gamification: GamificationState;
  ui: UIState;
}

export interface InterviewState {
  currentInterview: Interview | null;
  interviews: Interview[];
  loading: boolean;
  error: string | null;
}

export interface GamificationState {
  stats: UserStats | null;
  badges: Badge[];
  userBadges: UserBadge[];
  leaderboard: LeaderboardEntry[];
  games: Game[];
  loading: boolean;
  error: string | null;
}

export interface UIState {
  theme: 'light' | 'dark';
  language: 'es' | 'en' | 'pt' | 'fr';
  notifications: boolean;
  offlineMode: boolean;
}

// Service Types
export interface APIConfig {
  baseURL: string;
  timeout: number;
  headers: Record<string, string>;
}

// Utility Types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface PaginationParams {
  limit: number;
  offset: number;
}

