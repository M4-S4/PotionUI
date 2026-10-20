import { test, expect } from '@playwright/test';
import { loginAsOwner, ownerToken, screenshot, expectNoBoxOverlap } from './helpers';

const JOURNEY = 'admin-llm-detail-header';

const VIEWPORTS = [
	{ name: '1440', width: 1440, height: 900 },
	{ name: '390', width: 390, height: 844 }
];

const LONG_NAME = `e2e-hdr-${Date.now()}-a-very-long-llm-configuration-name-that-cannot-fit-on-a-phone`;

for (const viewport of VIEWPORTS) {
	test(`LLM configuration detail header keeps name, subtitle and actions apart at ${viewport.name}`, async ({
		page
	}) => {
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await loginAsOwner(page);
		const token = await ownerToken(page);
		const headers = { Authorization: `Bearer ${token}` };
		const create = await page.request.post('/api/llm/configurations', {
			headers,
			data: {
				name: LONG_NAME,
				type: 'ollama',
				enabled: true,
				base_url: 'http://127.0.0.1:1',
				model: 'llama3',
				system_message: 'You are a helpful assistant.'
			}
		});
		expect(create.ok(), `config create -> ${create.status()}`).toBeTruthy();
		const id = (await create.json()).data?.id;

		try {
			await page.goto('/admin?tab=llm');
			const row = page.getByRole('row').filter({ hasText: LONG_NAME });
			if (viewport.width < 640 && (await row.count()) === 0) {
				await page.getByText(LONG_NAME).locator('visible=true').first().click({ timeout: 15000 });
			} else {
				await row.click({ timeout: 15000 });
			}

			const title = page.locator('[data-detail-title]');
			await expect(title).toHaveText(LONG_NAME, { timeout: 15000 });
			const metrics = await title.evaluate((el) => {
				const style = getComputedStyle(el);
				return {
					lineHeight: parseFloat(style.lineHeight) || el.getBoundingClientRect().height,
					height: el.getBoundingClientRect().height,
					truncated: el.scrollWidth > el.clientWidth,
					overflow: style.textOverflow
				};
			});
			expect(metrics.overflow).toBe('ellipsis');
			expect(metrics.height).toBeLessThanOrEqual(metrics.lineHeight * 1.5);
			if (viewport.width < 640) {
				expect(metrics.truncated).toBe(true);
				const box = await title.boundingBox();
				expect(box!.width).toBeGreaterThan(120);
			}

			const subtitle = page.locator('[data-detail-subtitle]');
			await expect(subtitle).toContainText('ollama · llama3');
			const titleBox = await title.boundingBox();
			const subtitleBox = await subtitle.boundingBox();
			expect(subtitleBox!.y).toBeGreaterThanOrEqual(titleBox!.y + titleBox!.height - 1);

			await expectNoBoxOverlap([
				{ name: 'title', locator: title },
				{ name: 'subtitle', locator: subtitle },
				{ name: 'enabled toggle', locator: page.locator('[data-detail-actions] label') },
				{ name: 'duplicate', locator: page.getByRole('button', { name: 'Duplicate configuration' }) },
				{ name: 'delete', locator: page.getByRole('button', { name: 'Delete configuration' }) }
			]);

			await title.hover();
			await expect(page.getByText(LONG_NAME, { exact: true }).last()).toBeVisible();
			await screenshot(page, JOURNEY, `detail-header-${viewport.name}`);
		} finally {
			await page.request.delete(`/api/llm/configurations/${id}`, { headers });
		}
	});
}
