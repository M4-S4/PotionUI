import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const BELOW_FLOOR = /\btext-(2xs|xs)\b/;

function source(relative: string): string {
	return readFileSync(new URL(relative, import.meta.url), 'utf8');
}

function linesContaining(text: string, needle: string): string[] {
	return text.split('\n').filter((line) => line.includes(needle));
}

describe('the type floor on shared labels', () => {
	it('keeps the form field label at 12px', () => {
		const css = source('../../app.css');
		const rule = css.slice(css.indexOf('\t.label {'), css.indexOf('}', css.indexOf('\t.label {')));
		expect(rule).toMatch(/\btext-sm\b/);
		expect(rule).not.toMatch(BELOW_FLOOR);
	});

	it('keeps the folded workbench rail and header labels at 12px', () => {
		const panels = source('../../routes/generate/components/GenerationPanels.svelte');
		const labels = linesContaining(panels, '>Workbench</span>');
		expect(labels).toHaveLength(2);
		for (const line of labels) {
			expect(line).toMatch(/\btext-sm\b/);
			expect(line).not.toMatch(BELOW_FLOOR);
		}
		const expand = linesContaining(panels, '>Expand</span>');
		expect(expand).toHaveLength(1);
		expect(expand[0]).toMatch(/\btext-sm\b/);
	});

	it('keeps the floating workbench header label at 12px', () => {
		const floating = source('../../routes/generate/components/FloatingWorkbench.svelte');
		const labels = linesContaining(floating, '>Workbench</span>');
		expect(labels).toHaveLength(1);
		expect(labels[0]).toMatch(/\btext-sm\b/);
		expect(labels[0]).not.toMatch(BELOW_FLOOR);
	});

	it('leaves no sub-12px text in the shared Kbd, Pagination, SegmentedControl, UserPicker and PageTitle primitives', () => {
		for (const name of ['Kbd', 'Pagination', 'SegmentedControl', 'UserPicker', 'PageTitle']) {
			expect(source(`../components/ui/${name}.svelte`), name).not.toMatch(BELOW_FLOOR);
		}
	});

	it('drops the unused keybinding hint style', () => {
		expect(source('../../app.css')).not.toContain('keybinding-hint');
	});
});
