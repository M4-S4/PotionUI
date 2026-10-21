import { test, expect } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot, expectNoBoxOverlap } from './helpers';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'phone-chat-entry';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async () => {
	await fake.close();
});

for (const path of ['/generate', '/models', '/history', '/settings']) {
	test(`phone opens the chat sheet from the tab bar on ${path}`, async ({ page }) => {
		await page.setViewportSize({ width: 390, height: 844 });
		await loginAsOwner(page);
		const token = await ownerToken(page);
		await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);

		await page.goto(path);
		const nav = page.locator('nav.fixed.bottom-0');
		const chat = nav.getByRole('button', { name: 'Chat', exact: true });
		await expect(chat).toBeVisible({ timeout: 15000 });

		await expectNoBoxOverlap([
			{ name: 'generate tab', locator: nav.getByRole('link', { name: 'Generate' }) },
			{ name: 'history tab', locator: nav.getByRole('link', { name: 'History' }) },
			{ name: 'models tab', locator: nav.getByRole('link', { name: 'Models' }) },
			{ name: 'settings tab', locator: nav.getByRole('link', { name: 'Settings' }) },
			{ name: 'chat tab', locator: chat }
		]);

		await chat.click();
		const sheet = page.getByRole('dialog', { name: 'AI chat' });
		await expect(sheet).toBeVisible({ timeout: 15000 });
		await expect(sheet.locator('header.chat-header')).toBeVisible();
		await expect(page.getByRole('dialog', { name: 'AI chat' })).toHaveCount(1);
		await expect(page.locator('.chat-shell')).toHaveCount(1);
		await screenshot(page, JOURNEY, `sheet-${path.slice(1)}`);

		await page.keyboard.press('Escape');
		await expect(sheet).toHaveCount(0);
	});
}

test('phone Generate top-bar button opens the same single chat sheet', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await loginAsOwner(page);
	await page.goto('/generate');
	await page.getByRole('button', { name: 'Open AI chat' }).click();
	await expect(page.getByRole('dialog', { name: 'AI chat' })).toHaveCount(1, { timeout: 15000 });
	await expect(page.locator('.chat-shell')).toHaveCount(1);
});

test('desktop has no chat tab bar item', async ({ page }) => {
	await page.setViewportSize({ width: 1440, height: 900 });
	await loginAsOwner(page);
	await page.goto('/models');
	await expect(page.getByRole('button', { name: 'AI Chat' })).toBeVisible({ timeout: 15000 });
	await expect(page.locator('nav.fixed.bottom-0')).toBeHidden();
});
