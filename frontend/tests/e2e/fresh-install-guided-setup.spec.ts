import { test, expect, type Request } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { screenshot, shotPath } from './helpers';

const JOURNEY = 'fresh-install-guided-setup';
const RECIPE_NAME = process.env.E2E_FRESH_RECIPE_NAME ?? '';
const PASSWORD = 'fresh-owner-pw-7c1e44';

test.use({ viewport: { width: 1440, height: 900 } });

test('a fresh instance runs Guided setup from the real wizard', async ({ page }) => {
	test.setTimeout(240000);
	const calls: string[] = [];
	const record = (request: Request) => {
		const url = new URL(request.url());
		if (!url.pathname.startsWith('/api/')) return;
		const line = `${request.method()} ${url.pathname}`;
		if (calls[calls.length - 1] !== line) calls.push(line);
	};
	page.on('request', record);

	try {
		await page.goto('/');
		await expect(page.getByRole('heading', { name: 'Create the owner account' })).toBeVisible({ timeout: 20000 });
		await screenshot(page, JOURNEY, '01-claim-owner');

		await page.locator('#username').fill('fresh-owner');
		await page.locator('#email').fill('fresh-owner@example.com');
		await page.locator('#password').fill(PASSWORD);
		await page.locator('#confirmPassword').fill(PASSWORD);
		await page.getByRole('button', { name: 'Create owner account' }).click();

		await page.waitForURL(/\/setup/, { timeout: 20000 });
		await expect(page.getByRole('heading', { name: 'Models location' })).toBeVisible({ timeout: 20000 });
		await screenshot(page, JOURNEY, '02-models-location');

		await page.getByRole('button', { name: 'Skip' }).click();
		await expect(page.getByText(RECIPE_NAME)).toBeVisible({ timeout: 20000 });
		await screenshot(page, JOURNEY, '03-recipe-list');

		const card = page.locator('div').filter({ hasText: RECIPE_NAME }).filter({ has: page.getByRole('button', { name: 'Start' }) }).last();
		await card.getByRole('button', { name: 'Start' }).click();
		await expect(page.getByText('Guided setup', { exact: true })).toBeVisible({ timeout: 20000 });
		await screenshot(page, JOURNEY, '04-run-started');

		const step = (title: string) =>
			page
				.locator('div')
				.filter({ has: page.getByText(title, { exact: true }) })
				.filter({ hasText: /Done|Couldn't finish|Needs your go-ahead|Working on it|Not started yet/ })
				.last();

		await expect(step('Enable required plugins').getByText('Done', { exact: true })).toBeVisible({ timeout: 60000 });
		await expect(page.getByText("Couldn't finish")).toHaveCount(0);
		await expect(page.getByText('Plan required downloads wants to download')).toBeVisible({ timeout: 60000 });
		await screenshot(page, JOURNEY, '05-consent');
		await page.getByRole('button', { name: 'Approve and download' }).click();

		await expect(step('Download the Flux2 model files').getByText('Done', { exact: true })).toBeVisible({ timeout: 120000 });
		await screenshot(page, JOURNEY, '06-downloaded');
		await expect(step('Register downloaded models').getByText('Done', { exact: true })).toBeVisible({ timeout: 120000 });
		await expect(step('Install and assign the Flux2 preset').getByText('Done', { exact: true })).toBeVisible({ timeout: 120000 });
		await expect(page.getByText("Couldn't finish")).toHaveCount(0);
		await expect(page.getByText("You're all set")).toBeVisible({ timeout: 30000 });
		await screenshot(page, JOURNEY, '07-completed');

		await page.getByRole('button', { name: 'Refresh' }).click();
		const presetsCheck = page.locator('div').filter({ has: page.getByRole('heading', { name: 'Presets & models' }) }).last();
		await expect(presetsCheck.getByText('ready', { exact: true })).toBeVisible({ timeout: 30000 });
		await screenshot(page, JOURNEY, '08-presets-ready');

		await page.goto('/generate');
		await page.getByRole('button', { name: 'Choose a preset' }).click();
		await expect(page.getByText('Flux2', { exact: false }).first()).toBeVisible({ timeout: 15000 });
		await screenshot(page, JOURNEY, '09-generate');

		await page.setViewportSize({ width: 390, height: 844 });
		await screenshot(page, JOURNEY, '10-completed-390');
	} finally {
		const path = shotPath(JOURNEY, 'api-calls').replace(/\.png$/, '.txt');
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, calls.join('\n') + '\n');
		console.log(`API calls:\n${calls.join('\n')}`);
	}
});
