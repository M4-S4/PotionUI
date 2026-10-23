import { test, expect, type Route } from '@playwright/test';
import { loginAsOwner, screenshot } from './helpers';

const JOURNEY = 'organize-detected-tags';

const VIEWPORTS = [
	{ name: '1440', width: 1440, height: 900 },
	{ name: '390', width: 390, height: 844 }
];

const TAG_OPERATORS = ['has', 'has_not'];

const FACTS = [
	{ key: 'prompt', label: 'Prompt', subjects: ['generation'], kind: 'text', operators: ['contains', 'not_contains'], picker: { placeholder: 'a word or phrase' }, options: null, has_options_endpoint: false, description: '', source: 'core', component: null, previewable: true },
	{ key: 'tags', label: 'Tags', subjects: ['generation', 'upload', 'model'], kind: 'tag_list', operators: TAG_OPERATORS, picker: {}, options: null, has_options_endpoint: true, description: '', source: 'core', component: null, previewable: true },
	{ key: 'detected_tags', label: 'Detected tags', subjects: ['generation'], kind: 'tag_list', operators: TAG_OPERATORS, picker: {}, options: null, has_options_endpoint: true, description: '', source: 'core', component: null, previewable: true, triggers: ['detected_tags_changed'] }
];

const CATALOG = {
	subjects: [
		{ key: 'generation', label: 'Generations', trigger: 'generation_completed', trigger_label: 'When a new generation completes', collection_scope: 'history', supports_tags: true },
		{ key: 'upload', label: 'Library uploads', trigger: 'upload_created', trigger_label: 'When I upload a file', collection_scope: 'library', supports_tags: true },
		{ key: 'model', label: 'Models', trigger: 'model_added', trigger_label: 'When a model is added', collection_scope: 'models', supports_tags: false }
	],
	kinds: { tag_list: { operators: TAG_OPERATORS }, text: { operators: ['contains', 'not_contains'] } },
	operators: { has: 'has', has_not: 'does not have', contains: 'contains', not_contains: 'does not contain' },
	facts: FACTS,
	actions: [
		{ key: 'add_to_collection', label: 'Add to collection', subjects: ['generation', 'upload', 'model'], config_schema: [{ key: 'collection', kind: 'collection', label: 'Collection' }], requires_admin: false, undoable: true, source: 'core', component: null }
	]
};

const SUMMARY = {
	paused_by_admin: false,
	rule_cap: 50,
	rule_count: 0,
	hourly_limit: 500,
	subjects: {
		generation: { total: 0, active: 0, needs_attention: 0 },
		upload: { total: 0, active: 0, needs_attention: 0 },
		model: { total: 0, active: 0, needs_attention: 0 }
	}
};

const VOCABULARY = ['dog', 'grass', 'outdoors', 'solo'];

const json = (route: Route, data: unknown) => route.fulfill({ json: { success: true, data } });

for (const viewport of VIEWPORTS) {
	test(`a generation rule offers Detected tags with suggestions from what the tagger saw at ${viewport.name}`, async ({ page }) => {
		test.setTimeout(120000);
		await page.setViewportSize({ width: viewport.width, height: viewport.height });
		await loginAsOwner(page);

		const optionRequests: string[] = [];
		const previews: unknown[] = [];
		await page.route('**/api/organize/**', async (route) => {
			const url = new URL(route.request().url());
			const path = url.pathname.replace(/^.*\/api\/organize/, '');
			if (path === '/catalog') return json(route, CATALOG);
			if (path === '/summary') return json(route, SUMMARY);
			if (path === '/rules') return json(route, []);
			if (path === '/templates' || path === '/jobs') return json(route, []);
			if (path === '/activity') return json(route, { runs: [], next_before: null });
			if (path === '/facts/detected_tags/options') {
				optionRequests.push(url.searchParams.get('q') ?? '');
				const q = (url.searchParams.get('q') ?? '').toLowerCase();
				return json(route, VOCABULARY.filter((v) => v.includes(q)).map((v) => ({ value: v, label: v })));
			}
			if (path === '/preview') {
				previews.push(route.request().postDataJSON());
				return json(route, { matched: 12, already_handled: 0, would_change: 12, approximate: false, sample: [], duplicates: [] });
			}
			return route.fallback();
		});

		await page.goto('/auto-organize?subject=generations&rule=new');
		const nameInput = page.getByLabel('Rule name');
		await expect(nameInput).toBeVisible({ timeout: 15000 });
		await nameInput.fill('Dogs');

		await page.getByRole('button', { name: 'Add condition' }).click();
		const row = page.locator('[data-testid="condition-row"]').first();
		await row.locator('[aria-haspopup="listbox"]').first().click();
		await expect(page.getByTestId('rule-trigger-note')).toHaveCount(0);
		await page.getByRole('option', { name: 'Detected tags' }).click();
		await expect(page.getByTestId('rule-trigger-note')).toHaveText('Runs after Media Tagging finishes');

		const control = row.getByTestId('tag-names-input');
		await expect(control).toBeVisible();
		await control.getByLabel('Add a tag').fill('do');
		await expect(page.getByRole('option', { name: 'dog' })).toBeVisible();
		await expect(page.getByRole('option', { name: 'grass' })).toHaveCount(0);
		expect(optionRequests.length).toBeGreaterThan(0);
		await screenshot(page, JOURNEY, `${viewport.name}-1-suggestions`);

		await page.getByRole('option', { name: 'dog' }).click();
		await expect(control).toContainText('dog');
		await expect(page.getByTestId('preview-matched')).toHaveText('12', { timeout: 15000 });
		expect(JSON.stringify(previews[previews.length - 1])).toContain('"fact":"detected_tags"');
		await screenshot(page, JOURNEY, `${viewport.name}-2-condition-with-preview`);
	});
}
