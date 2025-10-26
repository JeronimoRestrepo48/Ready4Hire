/**
 * Interview Mutations
 * GraphQL mutations para entrevistas
 */

import { gql } from '@apollo/client';

export const START_INTERVIEW_MUTATION = gql`
  mutation StartInterview($input: StartInterviewInput!) {
    startInterview(input: $input) {
      id
      interviewId
      role
      mode
      status
      skillLevel
      totalQuestions
    }
  }
`;

export const PROCESS_ANSWER_MUTATION = gql`
  mutation ProcessAnswer($input: ProcessAnswerInput!) {
    processAnswer(input: $input) {
      id
      answerText
      score
      isCorrect
      emotion
      feedback
      timeTakenSeconds
      hintsUsed
    }
  }
`;

export const COMPLETE_INTERVIEW_MUTATION = gql`
  mutation CompleteInterview($interviewId: String!) {
    completeInterview(interviewId: $interviewId) {
      id
      averageScore
      successRate
      percentile
      strengths
      improvements
    }
  }
`;

