import { test, expect } from '@playwright/test';

/**
 * E2E Tests for landing/login experience
 */
test.describe('Landing Experience', () => {
  test('redirects unauthenticated users to login', async ({ page }) => {
    await page.goto('/');

    await expect(page).toHaveURL(/\/login/i);
    await expect(page.locator('.auth-title')).toContainText(/Ready4Hire/i);
    await expect(page.locator('input[type="email"]').first()).toBeVisible();
  });

  test('shows primary call to action on login', async ({ page }) => {
    await page.goto('/login');

    const loginButton = page.locator('button', { hasText: /Iniciar Sesión/i }).first();
    await expect(loginButton).toBeVisible();

    const registerTrigger = page.locator('button', { hasText: /Regístrate/i }).first();
    await expect(registerTrigger).toBeVisible();
  });

  test('remains accessible under responsive viewports', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/login');
    await expect(page.locator('.auth-container')).toBeVisible();

    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(page.locator('.auth-container')).toBeVisible();
  });
});

