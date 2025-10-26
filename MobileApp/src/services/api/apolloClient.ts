/**
 * Apollo Client Configuration
 * GraphQL client para comunicación con backend
 */

import { ApolloClient, InMemoryCache, HttpLink, split, from } from '@apollo/client';
import { getMainDefinition } from '@apollo/client/utilities';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';
import { setContext } from '@apollo/client/link/context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { onError } from '@apollo/client/link/error';

// URLs de la API
const HTTP_URL = process.env.GRAPHQL_URL || 'http://localhost:8001/graphql';
const WS_URL = process.env.WS_URL || 'ws://localhost:8001/graphql';

// Link HTTP
const httpLink = new HttpLink({
  uri: HTTP_URL,
});

// Link WebSocket para subscriptions
const wsLink = new GraphQLWsLink(
  createClient({
    url: WS_URL,
    connectionParams: async () => {
      const token = await AsyncStorage.getItem('auth_token');
      return {
        authorization: token ? `Bearer ${token}` : '',
      };
    },
  })
);

// Auth link para añadir JWT token
const authLink = setContext(async (_, { headers }) => {
  const token = await AsyncStorage.getItem('auth_token');
  
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : '',
    },
  };
});

// Error link para manejar errores globalmente
const errorLink = onError(({ graphQLErrors, networkError }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path }) => {
      console.error(
        `[GraphQL error]: Message: ${message}, Location: ${locations}, Path: ${path}`
      );
    });
  }
  
  if (networkError) {
    console.error(`[Network error]: ${networkError}`);
  }
});

// Split entre HTTP y WebSocket
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,
  from([errorLink, authLink, httpLink])
);

// Apollo Client instance
export const apolloClient = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache({
    typePolicies: {
      Query: {
        fields: {
          // Cache policies personalizadas
          interviews: {
            merge(existing = [], incoming) {
              return incoming;
            },
          },
        },
      },
    },
  }),
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
      errorPolicy: 'all',
    },
    query: {
      fetchPolicy: 'network-only',
      errorPolicy: 'all',
    },
    mutate: {
      errorPolicy: 'all',
    },
  },
});

export default apolloClient;

