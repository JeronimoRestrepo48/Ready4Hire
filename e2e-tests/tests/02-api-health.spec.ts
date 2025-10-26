import { test, expect } from '@playwright/test';

const API_BASE_URL = 'http://localhost:8001';

/**
 * E2E Tests for API Health Checks
 */
test.describe('API Health Checks', () => {
  
  test('Python backend health endpoint should respond', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v2/health`);
    
    // Should respond (200 or 503 if degraded)
    expect([200, 503]).toContain(response.status());
    
    const data = await response.json();
    expect(data).toHaveProperty('status');
  });

  test('Python backend root endpoint should respond', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/`);
    
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('message');
    expect(data).toHaveProperty('version');
  });

  test('API should have CORS headers', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v2/health`, {
      headers: {
        'Origin': 'http://localhost:5214'
      }
    });
    
    const headers = response.headers();
    expect(headers).toHaveProperty('access-control-allow-origin');
  });

  test('API should handle invalid endpoints gracefully', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v2/nonexistent`);
    
    expect(response.status()).toBe(404);
  });
});

