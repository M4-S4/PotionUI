import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), 'utf8');

const css = read('../../styles/generation-panel.css');
const panel = read('../GenerationPanel.svelte');
const thumb = read('./DeckThumb.svelte');
const tabBar = read('../MobileTabBar.svelte');
const tokens = read('../../styles/tokens.css');
const layout = read('../../../routes/+layout.svelte');

const PHONE = '@media (max-width: 767px) {';

function blockAt(source: string, start: number): string {
	const open = source.indexOf('{', start);
	let depth = 0;
	for (let index = open; index < source.length; index++) {
		if (source[index] === '{') depth++;
		if (source[index] === '}' && --depth === 0) return source.slice(open + 1, index);
	}
	throw new Error('unbalanced braces');
}

const phoneBlock = blockAt(css, css.indexOf(PHONE));

function phoneRule(selector: string): string {
	const escaped = selector.replace(/[.[\]()&]/g, '\\$&');
	const match = phoneBlock.match(new RegExp(`(?:^|\\n)\\s*${escaped} \\{([^}]*)\\}`));
	if (!match) throw new Error(`the phone block has no rule for ${selector}`);
	return match[1];
}

const px = (declarations: string, property: string): number => {
	const match = declarations.match(new RegExp(`(?:^|[\\s;])${property}: (-?\\d+)px`));
	if (!match) throw new Error(`no ${property} in "${declarations.trim()}"`);
	return Number(match[1]);
};

describe('deck on phones', () => {
	it('declares the phone block once, after the base rules it overrides', () => {
		expect(css.split(PHONE)).toHaveLength(2);
		const at = css.indexOf(PHONE);
		for (const selector of ['.generate-button', '.deck-title', '.deck-timing', '.deck-status', '.deck-ambient', '.deck']) {
			expect(css.lastIndexOf(`\n  ${selector} {`, at)).toBeGreaterThan(-1);
		}
	});

	it('lays the status block, transport and utilities out in two rows', () => {
		const deck = phoneRule('.deck');
		expect(deck).toContain('grid-template-columns: minmax(0, 1fr) auto');
		expect(deck).toContain('grid-template-areas: "status utilities" "transport transport"');
		expect(phoneRule('.deck-status')).toContain('grid-area: status');
		expect(phoneRule('.utility-cluster')).toContain('grid-area: utilities');
		expect(phoneRule('.run-cluster')).toContain('grid-area: transport');
	});

	it('adds up to the phone deck height: border, padding, both rows and the gap', () => {
		const deck = phoneRule('.deck');
		const rows = deck.match(/grid-template-rows: (\d+)px (\d+)px/)!;
		const padding = deck.match(/padding: (\d+)px (\d+)px (\d+)px/)!;
		const gap = deck.match(/gap: (\d+)px (\d+)px/)!;
		const total = 1 + Number(padding[1]) + Number(rows[1]) + Number(gap[1]) + Number(rows[2]) + Number(padding[3]);
		const declared = panel.match(/@media \(max-width: 767px\) \{\s*:global\(:root\) \{\s*--deck-height: (\d+)px/)!;
		expect(total).toBe(Number(declared[1]));
	});

	it('sits directly above the tab bar, at the one tab bar height token', () => {
		expect(phoneRule('&')).toContain('bottom: var(--tab-bar-height)');
		expect(tokens).toContain('--tab-bar-height: calc(4rem + env(safe-area-inset-bottom));');
		expect(tabBar).toContain('height: var(--tab-bar-height);');
	});

	it('keeps the eyebrow, timing readout and ambient halo out of the phone deck', () => {
		for (const selector of ['.deck-eyebrow', '.deck-timing', '.deck-ambient']) {
			expect(phoneRule(selector)).toContain('display: none');
		}
	});

	it('sets the title at 13px and keeps every control a 44px touch target or larger', () => {
		expect(px(phoneRule('.deck-title'), 'font-size')).toBe(13);
		const disc = phoneRule('.generate-button');
		expect(px(phoneRule('.generate-mark'), 'width')).toBeLessThan(px(disc, 'width'));
		const modeButton = css.match(/\n {2}\.mode-button \{([^}]*)\}/)![1];
		for (const declarations of [disc, phoneRule('.icon-button'), modeButton]) {
			expect(px(declarations, 'width')).toBeGreaterThanOrEqual(44);
			expect(px(declarations, 'height')).toBeGreaterThanOrEqual(44);
		}
	});

	it('shrinks the thumbnail tile to 44px and scales its icon and waveform', () => {
		const media = thumb.slice(thumb.indexOf(PHONE));
		expect(media).toMatch(/\.deck-thumb \{\s*width: 44px;\s*height: 44px;/);
		expect(media).toMatch(/\.deck-thumb :global\(\.deck-thumb-icon\) \{\s*width: 20px;\s*height: 20px;/);
		expect(media).toMatch(/\.deck-thumb-wave \{\s*width: 28px;\s*height: 16px;\s*margin-bottom: 14px;/);
	});
});

describe('deck clearance variables', () => {
	it('keeps the dock clearance equal to the deck height on desktop', () => {
		expect(panel).toMatch(/:global\(:root\) \{\s*--deck-height: 92px;\s*--dock-height: var\(--deck-height\);/);
	});

	it('clears the deck and the tab bar on phones', () => {
		expect(panel).toMatch(
			/@media \(max-width: 767px\) \{\s*:global\(:root\) \{\s*--deck-height: 140px;\s*--dock-height: calc\(var\(--deck-height\) \+ var\(--tab-bar-height\)\);/
		);
	});

	it('lifts the install banner above the deck only while the deck is mounted', () => {
		expect(panel).toMatch(/:global\(:root:has\(\.generation-panel\)\) \{\s*--tab-bar-lift: var\(--deck-height\);\s*--install-banner-z: 20;/);
		expect(layout).toContain('bottom-[calc(var(--tab-bar-height)_+_var(--tab-bar-lift,0px))]');
		expect(layout).toContain('z-[var(--install-banner-z,30)]');
		expect(layout).not.toMatch(/pathname[^\n]*generate/);
	});

	it('sizes the panel from its own height, not the clearance', () => {
		expect(css).toMatch(/\n {4}height: var\(--deck-height\);/);
		expect(css).not.toMatch(/height: var\(--dock-height\);/);
	});
});
