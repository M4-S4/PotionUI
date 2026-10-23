// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: ResolvedPromptPreview } = await import('../../src/lib/components/ResolvedPromptPreview.svelte');
const { createClassComponent } = await import('svelte/legacy');

const PROMPT = 'portrait, <<if $age > 65>>grey hair<<else if $age < 25>>youthful<<else>>auburn hair<<end>>, soft light';

let destroy: (() => void) | undefined;

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
});

async function mountOpen(variables: VariablesMap, prompt = PROMPT) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: ResolvedPromptPreview as never,
		target,
		props: { prompt, variables }
	});
	destroy = () => component.$destroy();
	target.querySelector<HTMLElement>('button[aria-expanded]')!.click();
	await tick();
	return target;
}

const age = (over: Record<string, unknown>): VariablesMap => ({
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18, ...over } as never
});

describe('ResolvedPromptPreview conditions', () => {
	it('shows one row per image with the rolled value, the branch and the highlighted text', async () => {
		const target = await mountOpen(age({}));
		const rows = Array.from(target.querySelectorAll('[data-testid="condition-preview-row"]'));
		expect(rows).toHaveLength(4);
		for (const row of rows) {
			const cells = row.querySelectorAll('td');
			const value = Number(cells[1].textContent!.match(/\d+/)![0]);
			const branch = value > 65 ? 'IF' : value < 25 ? 'ELSE IF' : 'OTHERWISE';
			const text = value > 65 ? 'grey hair' : value < 25 ? 'youthful' : 'auburn hair';
			expect(cells[2].textContent!.trim()).toBe(branch);
			expect(cells[3].querySelector('span')!.textContent).toBe(text);
			expect(cells[3].textContent!.replace(/\s+/g, ' ').trim()).toBe(`portrait, ${text}, soft light`);
		}
		expect(rows.map((r) => r.querySelector('td')!.textContent)).toEqual(['Image 1', 'Image 2', 'Image 3', 'Image 4']);
	});

	it('shows every branch and the note while the variable varies', async () => {
		const target = await mountOpen(age({}));
		expect(target.textContent).toContain('portrait, grey hair / youthful / auburn hair, soft light');
		expect(target.querySelector('[aria-label="Variables that vary"]')!.textContent).toContain(
			'age re-rolls per image — the condition follows it'
		);
		expect(target.querySelector('[data-testid="resolved-row-note"]')!.textContent).toContain('the condition follows it');
	});

	it('shows only the chosen text and no table when every variable read is settled', async () => {
		const target = await mountOpen(age({ mode: 'pin', value: 70 }));
		expect(target.querySelector('[data-testid="condition-preview-table"]')).toBeNull();
		expect(target.querySelector('[aria-label="Variables that vary"]')).toBeNull();
		expect(target.textContent).toContain('portrait, grey hair, soft light');
		expect(target.textContent).not.toContain(' / ');
		expect(target.textContent).not.toContain('<<');
	});

	it('uses the otherwise branch for a settled value that matches nothing', async () => {
		const target = await mountOpen(age({ mode: 'pin', value: 40 }));
		expect(target.textContent).toContain('portrait, auburn hair, soft light');
	});

	it('collapses a shuffle number to a single row', async () => {
		const target = await mountOpen(age({ mode: 'shuffle' }));
		const rows = target.querySelectorAll('[data-testid="condition-preview-row"]');
		expect(rows).toHaveLength(1);
		expect(rows[0].querySelector('td')!.textContent).toBe('All 4');
	});
});
