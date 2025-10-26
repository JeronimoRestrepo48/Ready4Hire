/**
 * Interview Queries
 * GraphQL queries para entrevistas
 */

import { gql } from '@apollo/client';

export const GET_ACTIVE_INTERVIEW = gql`
  query GetActiveInterview($interviewId: String!) {
    interviewDetails(interviewId: $interviewId) {
      interview {
        id
        interviewId
        role
        mode
        status
        currentPhase
        averageScore
        totalQuestions
        correctAnswers
      }
      questions {
        id
        text
        category
        difficulty
        topic
        expectedConcepts
      }
      answers {
        id
        answerText
        score
        isCorrect
        emotion
        feedback
      }
    }
  }
`;

export const GET_USER_INTERVIEWS = gql`
  query GetUserInterviews($userId: Int!, $status: String, $limit: Int) {
    interviews(userId: $userId, status: $status, limit: $limit) {
      id
      interviewId
      role
      mode
      status
      createdAt
      completedAt
      averageScore
      totalQuestions
      correctAnswers
    }
  }
`;

