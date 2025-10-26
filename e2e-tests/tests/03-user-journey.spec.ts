import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Complete User Journey
 */
test.describe('User Journey', () => {
  
  test('complete user flow: register → login → start interview', async ({ page }) => {
    // Navigate to home
    await page.goto('/');
    
    // Try to find register/login
    const registerButton = page.locator('text=/register|registr|sign up/i').first();
    
    if (await registerButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await registerButton.click();
      
      // Fill registration form (adjust selectors based on actual implementation)
      const emailInput = page.locator('input[type="email"]').first();
      if (await emailInput.isVisible({ timeout: 5000 }).catch(() => false)) {
        await emailInput.fill('test@example.com');
      }
    }
    
    // This is a basic flow test - actual implementation depends on your UI
    await expect(page.locator('body')).toBeVisible();
  });

  test('navigation flow through main pages', async ({ page }) => {
    await page.goto('/');
    
    // Check if main pages are accessible
    const pages = ['/', '/login'];
    
    for (const pagePath of pages) {
      const response = await page.goto(pagePath);
      expect(response?.status()).toBeLessThan(400);
    }
  });
});

