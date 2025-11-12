import { test, expect } from '@playwright/test';

/**
 * E2E Tests for unauthenticated user journey
 */
test.describe('User Journey', () => {
  test('login form renders required fields', async ({ page }) => {
    await page.goto('/login');

    await expect(page.locator('h1.auth-title')).toContainText(/Ready4Hire/i);
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('button', { hasText: /Iniciar Sesión/i })).toBeVisible();
  });

  test('registration wizard entry point is available', async ({ page }) => {
    await page.goto('/login');

    const registerButton = page.locator('button', { hasText: /Regístrate/i }).first();
    await expect(registerButton).toBeVisible();

    await registerButton.click();
    await expect(page.locator('.progress-text')).toContainText(/Paso 1/i);
    await expect(page.locator('input[type="email"]')).toBeVisible();
  });

  test('public routes respond without error', async ({ page }) => {
    const publicPaths = ['/', '/login'];

    for (const path of publicPaths) {
      const response = await page.goto(path);
      expect(response?.status()).toBeLessThan(400);
    }
  });
});

