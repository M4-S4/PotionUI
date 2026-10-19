import { test, expect, type Page } from '@playwright/test';
import { screenshot } from './helpers';

const JOURNEY = 'register-email-error';

const VIEWPORTS = [
	{ label: '1440', width: 1440, height: 900 },
	{ label: '390', width: 390, height: 844 }
] as const;

const EMAIL_ERROR = 'Value is not a valid email address: An email address must have an @-sign.';

async function mockOpenInstance(page: Page) {
	await page.route('**/api/setup/status', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ needs_owner: false, registration_open: true })
		});
	});
}

async function fillForm(page: Page, email: string) {
	await page.goto('/register');
	await page.locator('#username').fill('owner');
	await page.locator('#email').fill(email);
	await page.locator('#password').fill('longenough1');
	await page.locator('#confirmPassword').fill('longenough1');
}

for (const viewport of VIEWPORTS) {
	test.describe(`register email (${viewport.label})`, () => {
		test.use({ viewport: { width: viewport.width, height: viewport.height } });

		test('shows the server reason under the email field', async ({ page }) => {
			await mockOpenInstance(page);
			await page.route('**/api/auth/register', async (route) => {
				await route.fulfill({
					status: 422,
					contentType: 'application/json',
					headers: { 'access-control-allow-origin': '*' },
					body: JSON.stringify({
						success: false,
						error: 'validation_error',
						message: 'Request validation failed',
						detail: [{ type: 'value_error', loc: ['body', 'email'], msg: EMAIL_ERROR.replace('Value', 'value') }]
					})
				});
			});

			await fillForm(page, 'owner@badhost');
			await page.getByRole('button', { name: 'Create Account' }).click();

			const message = page.locator('#email-error');
			await expect(message).toContainText('not a valid email address');
			await expect(page.locator('#email')).toHaveAttribute('aria-invalid', 'true');
			await expect(page.locator('#username')).toHaveValue('owner');
			await expect(page.locator('#email')).toHaveValue('owner@badhost');
			await expect(page.locator('#password')).toHaveValue('longenough1');
			await screenshot(page, JOURNEY, `${viewport.label}-rejected`);
		});

		test('accepts a .local address', async ({ page }) => {
			let sentEmail = '';
			await mockOpenInstance(page);
			await page.route('**/api/auth/register', async (route) => {
				sentEmail = route.request().postDataJSON().email;
				await route.fulfill({
					status: 200,
					contentType: 'application/json',
					headers: { 'access-control-allow-origin': '*' },
					body: JSON.stringify({
						success: true,
						message: 'User registered successfully',
						data: {
							access_token: 'a.b.c',
							token_type: 'bearer',
							user: {
								id: 'u1',
								username: 'owner',
								email: 'admin@potionui.local',
								account_type: 'ADMIN',
								created_at: null,
								last_login: null,
								avatar_url: null,
								has_local_password: true
							}
						}
					})
				});
			});

			await fillForm(page, 'admin@potionui.local');
			await page.getByRole('button', { name: 'Create Account' }).click();

			await expect.poll(() => sentEmail).toBe('admin@potionui.local');
			await expect(page.locator('#email-error')).toHaveCount(0);
			await screenshot(page, JOURNEY, `${viewport.label}-accepted`);
		});
	});
}
