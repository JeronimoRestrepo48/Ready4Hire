/**
 * Interview Context
 * Manejo de entrevistas activas
 */

import React, { createContext, useState, useContext, ReactNode } from 'react';
import { useMutation, useQuery } from '@apollo/client';
import { 
  START_INTERVIEW_MUTATION, 
  PROCESS_ANSWER_MUTATION,
  COMPLETE_INTERVIEW_MUTATION 
} from '../graphql/mutations/interview';
import { GET_ACTIVE_INTERVIEW } from '../graphql/queries/interview';

interface Question {
  id: string;
  text: string;
  category: string;
  difficulty: string;
  topic: string;
}

interface Interview {
  id: string;
  role: string;
  mode: string;
  status: string;
  currentQuestion?: Question;
  totalQuestions: number;
  answeredQuestions: number;
}

interface InterviewContextType {
  activeInterview: Interview | null;
  isLoading: boolean;
  startInterview: (userId: number, role: string, mode: string, skillLevel: string) => Promise<void>;
  processAnswer: (interviewId: string, answerText: string, timeTaken: number) => Promise<any>;
  completeInterview: (interviewId: string) => Promise<any>;
  refreshInterview: () => void;
}

const InterviewContext = createContext<InterviewContextType | undefined>(undefined);

export const InterviewProvider = ({ children }: { children: ReactNode }) => {
  const [activeInterview, setActiveInterview] = useState<Interview | null>(null);

  const [startInterviewMutation, { loading: startLoading }] = useMutation(START_INTERVIEW_MUTATION);
  const [processAnswerMutation, { loading: answerLoading }] = useMutation(PROCESS_ANSWER_MUTATION);
  const [completeInterviewMutation, { loading: completeLoading }] = useMutation(COMPLETE_INTERVIEW_MUTATION);

  const { refetch } = useQuery(GET_ACTIVE_INTERVIEW, {
    skip: !activeInterview?.id,
    variables: { interviewId: activeInterview?.id },
  });

  const isLoading = startLoading || answerLoading || completeLoading;

  const startInterview = async (
    userId: number,
    role: string,
    mode: string,
    skillLevel: string
  ) => {
    try {
      const { data } = await startInterviewMutation({
        variables: {
          input: {
            userId,
            role,
            interviewType: 'technical',
            mode,
            skillLevel,
          },
        },
      });

      if (data?.startInterview) {
        setActiveInterview(data.startInterview);
      }
    } catch (error) {
      console.error('Start interview error:', error);
      throw error;
    }
  };

  const processAnswer = async (
    interviewId: string,
    answerText: string,
    timeTaken: number = 0
  ) => {
    try {
      const { data } = await processAnswerMutation({
        variables: {
          input: {
            interviewId,
            answerText,
            timeTakenSeconds: timeTaken,
          },
        },
      });

      return data?.processAnswer;
    } catch (error) {
      console.error('Process answer error:', error);
      throw error;
    }
  };

  const completeInterview = async (interviewId: string) => {
    try {
      const { data } = await completeInterviewMutation({
        variables: { interviewId },
      });

      setActiveInterview(null);
      return data?.completeInterview;
    } catch (error) {
      console.error('Complete interview error:', error);
      throw error;
    }
  };

  const refreshInterview = () => {
    if (refetch) {
      refetch();
    }
  };

  return (
    <InterviewContext.Provider
      value={{
        activeInterview,
        isLoading,
        startInterview,
        processAnswer,
        completeInterview,
        refreshInterview,
      }}
    >
      {children}
    </InterviewContext.Provider>
  );
};

export const useInterview = () => {
  const context = useContext(InterviewContext);
  if (context === undefined) {
    throw new Error('useInterview must be used within an InterviewProvider');
  }
  return context;
};

