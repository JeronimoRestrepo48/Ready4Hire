/**
 * Auth Mutations
 * GraphQL mutations para autenticación
 */

import { gql } from '@apollo/client';

export const LOGIN_MUTATION = gql`
  mutation Login($input: LoginInput!) {
    login(input: $input) {
      token
      user {
        id
        email
        name
        lastName
        level
        experience
        totalPoints
      }
    }
  }
`;

export const REGISTER_MUTATION = gql`
  mutation Register($input: RegisterInput!) {
    register(input: $input) {
      id
      email
      name
      lastName
    }
  }
`;

