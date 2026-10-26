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

const COMMENT_OPEN = '/' + '*';
const COMMENT_CLOSE = '*' + '/';
const FLOOR = 12;

function withoutComments(css: string): string {
	let out = '';
	let at = 0;
	for (let open = css.indexOf(COMMENT_OPEN); open !== -1; open = css.indexOf(COMMENT_OPEN, at)) {
		out += css.slice(at, open);
		at = css.indexOf(COMMENT_CLOSE, open) + COMMENT_CLOSE.length;
	}
	return out + css.slice(at);
}

interface Rule {
	selectors: string[];
	body: string;
}

function rulesOf(css: string): Rule[] {
	return [...withoutComments(css).matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((rule) => ({
		selectors: rule[1].split(',').map((entry) => entry.trim()),
		body: rule[2]
	}));
}

function sizeOf(body: string): number | null {
	const size = body.match(/font-size:\s*([\d.]+)px/) ?? body.match(/font:\s*[^;]*?([\d.]+)px/);
	return size ? parseFloat(size[1]) : null;
}

function declaredSizes(css: string, selector: string): number[] {
	return rulesOf(css)
		.filter((rule) => rule.selectors.includes(selector))
		.map((rule) => sizeOf(rule.body))
		.filter((size): size is number => size !== null);
}

function bodyOf(css: string, selector: string): string {
	return rulesOf(css)
		.filter((rule) => rule.selectors.includes(selector))
		.map((rule) => rule.body)
		.join(' ');
}

const KEYBOARD_HINTS = ['.esc'];
const HINT_FLOOR = 11;

describe('the type floor in the segment composer stylesheet', () => {
	const css = source('./segment-composer.css');

	it.each([
		'.add-segment',
		'.resolved-title',
		'.resolved-stats',
		'.resolved-body',
		'.menu-item',
		'.menu-item .shortcut',
		'.menu-glyph',
		'.chip-tooltip',
		'.finder-heading span',
		'.index'
	])('never declares %s below 12px', (selector) => {
		const sizes = declaredSizes(css, selector);
		expect(sizes.length).toBeGreaterThan(0);
		for (const size of sizes) expect(size).toBeGreaterThanOrEqual(FLOOR);
	});

	it('leaves no selector whose last declared size is below 12px, bar the keyboard hints', () => {
		const last = new Map<string, number>();
		for (const rule of rulesOf(css)) {
			const size = sizeOf(rule.body);
			if (size === null) continue;
			for (const selector of rule.selectors) last.set(selector, size);
		}
		const below = [...last].filter(([, size]) => size < FLOOR).map(([selector]) => selector);
		expect(below).toEqual(KEYBOARD_HINTS);
		for (const hint of KEYBOARD_HINTS) expect(last.get(hint)).toBeGreaterThanOrEqual(HINT_FLOOR);
	});

	it('only exempts a class that renders as a key hint', () => {
		const dropdown = source('../components/AutocompleteDropdown.svelte');
		expect(dropdown).toContain('<span class="esc">Esc</span>');
	});

	it('keeps the resolved stats in the mono tabular idiom of the other counts', () => {
		const body = bodyOf(css, '.resolved-stats');
		expect(body).toMatch(/font:\s*400 12px "IBM Plex Mono"/);
		expect(body).toMatch(/font-variant-numeric:\s*tabular-nums/);
	});

	it('lays the resolved toggle out inline, chevron beside its title', () => {
		const body = bodyOf(css, '.resolved-toggle');
		expect(body).toMatch(/display:\s*flex/);
		expect(body).toMatch(/align-items:\s*center/);
	});

	it('no longer carries the dead current-value rules', () => {
		expect(css).not.toContain('current-value');
	});

	it('reads both the font-size property and the font shorthand, in file order', () => {
		const sample = '.a { font-size: 9px; }\n.b { color: red; }\n.a, .c { font: 400 7.5px "Plex"; }';
		expect(declaredSizes(sample, '.a')).toEqual([9, 7.5]);
		expect(declaredSizes(sample, '.c')).toEqual([7.5]);
		expect(declaredSizes(sample, '.b')).toEqual([]);
	});
});
