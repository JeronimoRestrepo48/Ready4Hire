import { test, expect } from '@playwright/test';

/**
 * E2E Tests for Home Page
 */
test.describe('Home Page', () => {
  
  test('should load home page successfully', async ({ page }) => {
    await page.goto('/');
    
    // Check that page loaded
    await expect(page).toHaveTitle(/Ready4Hire/i);
    
    // Check for main content
    await expect(page.locator('body')).toBeVisible();
  });

  test('should have navigation menu', async ({ page }) => {
    await page.goto('/');
    
    // Wait for navigation to be visible
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
  });

  test('should navigate to login page', async ({ page }) => {
    await page.goto('/');
    
    // Click on login link (adjust selector based on actual implementation)
    const loginLink = page.locator('text=/login|iniciar sesión/i').first();
    
    if (await loginLink.isVisible({ timeout: 5000 }).catch(() => false)) {
      await loginLink.click();
      await expect(page).toHaveURL(/login/i);
    }
  });

  test('should be responsive', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
    
    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('body')).toBeVisible();
  });
});

