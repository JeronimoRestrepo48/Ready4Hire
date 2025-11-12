import { test, expect } from '@playwright/test';

const API_BASE_URL = 'http://localhost:8001';

/**
 * E2E Performance Tests
 */
test.describe('Performance', () => {
  test('home page should load within 3 seconds', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/login');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(3000);
  });

  test('API health check should respond quickly', async ({ request }) => {
    const startTime = Date.now();
    await request.get(`${API_BASE_URL}/api/v2/health`);
    const responseTime = Date.now() - startTime;

    expect(responseTime).toBeLessThan(2000);
  });

  test('page should not have excessive network requests', async ({ page }) => {
    const requests: any[] = [];

    page.on('request', request => requests.push(request));

    await page.goto('/login');
    await page.waitForLoadState('networkidle');

    // Should not make more than 50 requests
    expect(requests.length).toBeLessThan(50);
  });
});

