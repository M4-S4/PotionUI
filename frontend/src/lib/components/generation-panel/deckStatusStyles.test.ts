import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';

const css = readFileSync(new URL('../../styles/generation-panel.css', import.meta.url), 'utf8');

function rule(selector: string): string {
	const escaped = selector.replace(/[.[\]()]/g, '\\$&');
	const match = css.match(new RegExp(`\\n\\s*${escaped} \\{([^}]*)\\}`));
	if (!match) throw new Error(`generation-panel.css has no rule for ${selector}`);
	return match[1];
}

const px = (declarations: string, property: string): number => {
	const match = declarations.match(new RegExp(`(?:^|[\\s;])${property}: (-?\\d+)px`));
	if (!match) throw new Error(`no ${property} in "${declarations.trim()}"`);
	return Number(match[1]);
};

describe('deck copy left edge', () => {
	it('hangs the dot out of the text flow', () => {
		expect(rule('.deck-title-line')).toContain('position: relative');
		const dot = rule('.deck-dot');
		expect(dot).toContain('position: absolute');
		expect(px(dot, 'margin-left')).toBeLessThan(0);
	});

	it('indents none of the three copy lines', () => {
		for (const selector of ['.deck-copy', '.deck-eyebrow', '.deck-title-line', '.deck-title', '.deck-detail']) {
			expect(rule(selector)).not.toMatch(/(?:^|[\s;])(?:padding|margin)(?:-left)?:/);
		}
	});

	it('keeps the hanging dot clear of both the thumbnail and the text', () => {
		const dot = rule('.deck-dot');
		const hang = -px(dot, 'margin-left');
		const gutter = px(rule('.deck-status'), 'gap');
		expect(hang - px(dot, 'width')).toBeGreaterThanOrEqual(5);
		expect(gutter - hang).toBeGreaterThanOrEqual(5);
	});
});

describe('deck dot colour', () => {
	it('paints the dot and its halo from one status token', () => {
		const dot = rule('.deck-dot');
		expect(dot).toContain('background: rgb(var(--status))');
		expect(dot).toContain('box-shadow: 0 0 0 3px rgb(var(--status) / .16)');
		expect(dot).not.toMatch(/#[0-9a-f]{3,8}\b/i);
	});

	it('is muted by default and takes each state token from its tone class', () => {
		expect(rule('.deck-dot')).toContain('--status: var(--fg-subtle)');
		const tokens = { success: '--success', signal: '--signal', warning: '--warning', danger: '--danger' };
		for (const [tone, token] of Object.entries(tokens)) expect(rule(`.deck-dot.is-${tone}`)).toContain(`--status: var(${token})`);
	});
});
