import { test, expect } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot, expectNoBoxOverlap } from './helpers';
import { startFakeLLM, seedFakeLlmConfig, type FakeLLMServer } from './fake-llm';

const JOURNEY = 'chat-header-phone';
const BACKEND = process.env.E2E_BACKEND_URL || 'http://127.0.0.1:8055';

const VIEWPORTS = [
	{ name: '1440', width: 1440, height: 900 },
	{ name: '390', width: 390, height: 844 }
];

let fake: FakeLLMServer;

test.beforeEach(async () => {
	fake = await startFakeLLM();
});

test.afterEach(async () => {
	await fake.close();
});

for (const viewport of VIEWPORTS) {
	test(`chat header keeps title, mode selector and actions apart at ${viewport.name}`, async ({ page }) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await loginAsOwner(page);
		const token = await ownerToken(page);
		await seedFakeLlmConfig(page.request, BACKEND, token, fake.url);

		const phone = viewport.width < 768;
		await page.goto(phone ? '/generate' : '/models');
		const fab = page.getByRole('button', { name: phone ? 'Open AI chat' : 'AI Chat' });
		await expect(fab).toBeVisible({ timeout: 15000 });
		await fab.click();

		const header = page.locator('header.chat-header');
		await expect(header).toBeVisible({ timeout: 15000 });
		await expect(header.getByTestId('chat-mode-selector-trigger')).toBeVisible();

		await expectNoBoxOverlap([
			{ name: 'history toggle', locator: header.getByRole('button', { name: /^(Hide|Show) history$/ }) },
			{ name: 'new chat', locator: header.getByRole('button', { name: 'New chat' }) },
			{ name: 'title', locator: header.locator('.header-title') },
			{ name: 'mode selector', locator: header.getByTestId('chat-mode-selector-trigger') },
			{ name: 'context meter', locator: header.locator('.context-meter') },
			{ name: 'model button', locator: header.getByTestId('chat-header-model-trigger') },
			{ name: 'conversation options', locator: header.getByRole('button', { name: 'Conversation options' }) },
			{ name: 'close', locator: header.getByRole('button', { name: 'Close chat' }) }
		]);

		const box = await header.boundingBox();
		expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width + 0.5);
		const chip = await header.getByTestId('chat-mode-selector-trigger').boundingBox();
		expect(chip!.x + chip!.width).toBeLessThanOrEqual(viewport.width + 0.5);

		await screenshot(page, JOURNEY, `header-${viewport.name}`);
	});
}
