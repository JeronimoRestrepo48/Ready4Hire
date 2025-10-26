import { test, expect } from '@playwright/test';

/**
 * E2E Performance Tests
 */
test.describe('Performance', () => {
  
  test('home page should load within 3 seconds', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;
    
    expect(loadTime).toBeLessThan(3000);
  });

  test('API health check should respond quickly', async ({ request }) => {
    const startTime = Date.now();
    await request.get('http://localhost:8000/api/v2/health');
    const responseTime = Date.now() - startTime;
    
    expect(responseTime).toBeLessThan(2000);
  });

  test('page should not have excessive network requests', async ({ page }) => {
    const requests: any[] = [];
    
    page.on('request', request => requests.push(request));
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Should not make more than 50 requests
    expect(requests.length).toBeLessThan(50);
  });
});

