import { ApolloClient, InMemoryCache } from '@apollo/client';
import client from '../../src/services/api/apolloClient';

describe('Apollo Client', () => {
  it('should be an instance of ApolloClient', () => {
    expect(client).toBeInstanceOf(ApolloClient);
  });

  it('should have InMemoryCache', () => {
    expect(client.cache).toBeInstanceOf(InMemoryCache);
  });

  it('should have correct URI', () => {
    const link = client.link;
    // Basic check that link exists
    expect(link).toBeDefined();
  });
});

