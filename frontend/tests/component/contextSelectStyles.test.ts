import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import postcss from 'postcss';

const css = readFileSync(resolve(process.cwd(), 'src/lib/styles/context-select.css'), 'utf8');
const appCss = readFileSync(resolve(process.cwd(), 'src/app.css'), 'utf8');
const root = postcss.parse(css);

function rules(selector: string): postcss.Rule[] {
	const found: postcss.Rule[] = [];
	root.walkRules((rule) => {
		if (rule.selectors.includes(selector)) found.push(rule);
	});
	return found;
}

function declaration(selector: string, prop: string): string | undefined {
	return rules(selector)
		.flatMap((rule) => rule.nodes.filter((node): node is postcss.Declaration => node.type === 'decl' && node.prop === prop))
		.at(-1)?.value;
}

const FIELD = '.ctx-select:not(.preset-picker-cta)';
const KEYBOARD = ":where(html:not([data-input='pointer']))";

describe('context select styles', () => {
	it('is part of the app stylesheet', () => {
		expect(appCss).toContain("@import './lib/styles/context-select.css';");
	});

	it('draws a 1px line-hover rule at rest and never a fill', () => {
		expect(declaration(`${FIELD}::before`, 'height')).toBe('1px');
		expect(declaration(`${FIELD}::before`, 'background')).toBe('rgb(var(--line-hover))');
		for (const selector of ['.ctx-select', FIELD, '.ctx-select-session:not(.preset-picker-cta)', '.ctx-select:focus-visible', '.ctx-seg']) {
			expect(declaration(selector, 'background'), selector).toBeUndefined();
			expect(declaration(selector, 'background-color'), selector).toBeUndefined();
			expect(declaration(selector, 'border-radius'), selector).toBeUndefined();
		}
	});

	it('thickens to a 2px signal rule on open and on keyboard focus', () => {
		expect(declaration(`${FIELD}::after`, 'height')).toBe('2px');
		expect(declaration(`${FIELD}::after`, 'background')).toBe('rgb(var(--signal))');
		expect(declaration(`${FIELD}::after`, 'transform')).toBe('scaleX(0)');
		expect(declaration('.ctx-select.is-open::after', 'transform')).toBe('scaleX(1)');
		expect(declaration(`${KEYBOARD} .ctx-select:focus-visible::after`, 'transform')).toBe('scaleX(1)');
		expect(declaration('.ctx-select:focus-visible::after', 'transform')).toBeUndefined();
	});

	it('keeps hover neutral, not signal', () => {
		expect(declaration(`${FIELD}:hover:not(:disabled)::before`, 'background')).toBe('rgb(var(--fg-subtle))');
	});

	it('keeps the rule out of the pointer path and the box size fixed', () => {
		expect(declaration(`${FIELD}::before`, 'pointer-events')).toBe('none');
		expect(declaration(`${FIELD}::after`, 'pointer-events')).toBe('none');
		expect(declaration(FIELD, 'height')).toBe('20px');
		expect(declaration('.ctx-select.is-open', 'height')).toBeUndefined();
		expect(declaration('.ctx-select:focus-visible', 'height')).toBeUndefined();
	});

	it('puts the label above the value, 12px at least, turning signal with the select', () => {
		expect(declaration('.ctx-step', 'flex-direction')).toBe('column');
		expect(Number.parseFloat(declaration('.ctx-label', 'font-size') ?? '0')).toBeGreaterThanOrEqual(12);
		const lit = rules('.ctx-step:has(.ctx-select.is-open) .ctx-label').concat(rules(`${KEYBOARD} .ctx-step:has(.ctx-select:focus-visible) .ctx-label`));
		expect(lit).toHaveLength(2);
		for (const rule of lit) expect(rule.nodes.some((node) => node.type === 'decl' && node.prop === 'color' && node.value === 'rgb(var(--signal))')).toBe(true);
	});

	it('keeps the global focus ring and only moves it 1px so it clears the label', () => {
		expect(declaration('.ctx-select:focus-visible', 'outline-offset')).toBe('1px');
		expect(declaration('.ctx-select:focus-visible', 'outline')).toBeUndefined();
		expect(declaration('.ctx-select:focus-visible', 'outline-style')).toBeUndefined();
	});

	it('leaves the Choose a preset call to action its own box and no rule', () => {
		expect(declaration('.ctx-select.preset-picker-cta', 'height')).toBe('32px');
		expect(declaration('.ctx-select.preset-picker-cta', 'border-radius')).toBe('4px');
		expect(declaration('.ctx-step:has(.preset-picker-cta)', 'flex-direction')).toBe('row');
		expect(declaration('.ctx-step:has(.preset-picker-cta)', 'margin-bottom')).toBe('0');
		root.walkRules((rule) => {
			if (rule.selectors.some((selector) => selector.includes('::before') || selector.includes('::after'))) {
				expect(rule.selectors.every((selector) => selector.includes(':not(.preset-picker-cta)') || selector.includes('.is-open') || selector.includes(':focus-visible')), rule.selector).toBe(true);
			}
		});
	});

	it('uses semantic tokens only', () => {
		expect(css).not.toMatch(/#[0-9a-fA-F]{3,8}\b/);
		expect(css).not.toMatch(/\b(zinc|gray)-\d/);
		expect(css).toMatch(/rgb\(var\(--/);
	});

	it('shares one bottom baseline for the rule, Save and Reload', () => {
		expect(declaration('.ctx-seg', 'align-items')).toBe('flex-end');
		expect(declaration('.ctx-seg', 'height')).toBe(declaration(FIELD, 'height'));
	});

	it('lifts the label, gap and select block so its rule lands on the tools baseline and the label keeps room above', () => {
		const px = (value: string | undefined) => Number.parseFloat(value ?? 'NaN');
		const BAR_CONTENT = 47;
		const TOOLS = 32;
		const label = px(declaration('.ctx-label', 'line-height'));
		const gap = px(declaration('.ctx-step', 'gap'));
		const field = px(declaration(FIELD, 'height'));
		const lift = px(declaration('.ctx-step', 'margin-bottom'));
		const block = label + gap + field;
		const top = (BAR_CONTENT - (block + lift)) / 2;
		const ruleBottom = top + block;
		const toolsBottom = (BAR_CONTENT + TOOLS) / 2;
		expect(ruleBottom).toBe(toolsBottom);
		expect(top).toBeGreaterThanOrEqual(5);
		expect(BAR_CONTENT - ruleBottom - top).toBeLessThanOrEqual(2.5);
	});

	it('keeps the rule on the tools baseline when the fit hides the labels', () => {
		const px = (value: string | undefined) => Number.parseFloat(value ?? 'NaN');
		const BAR_CONTENT = 47;
		const TOOLS = 32;
		const label = px(declaration('.ctx-label', 'line-height'));
		const gap = px(declaration('.ctx-step', 'gap'));
		const field = px(declaration(FIELD, 'height'));
		const lift = px(declaration('.ctx-step', 'margin-bottom'));
		const reserved = px(declaration('.ctx-step', 'min-height'));
		expect(reserved).toBe(label + gap + field);
		expect(declaration('.ctx-step', 'justify-content')).toBe('flex-end');
		const toolsBottom = (BAR_CONTENT + TOOLS) / 2;
		for (const content of [label + gap + field, field]) {
			const height = Math.max(reserved, content);
			const ruleBottom = (BAR_CONTENT - (height + lift)) / 2 + height;
			expect(ruleBottom, `content ${content}`).toBe(toolsBottom);
		}
	});

	it('leaves the call to action step without the reserved label slot', () => {
		expect(declaration('.ctx-step:has(.preset-picker-cta)', 'min-height')).toBe('0');
	});
});
