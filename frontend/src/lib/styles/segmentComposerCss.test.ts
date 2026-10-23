import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

function readCss(): string {
	return readFileSync(new URL('./segment-composer.css', import.meta.url), 'utf8');
}

const COMMENT_OPEN = '/' + '*';
const COMMENT_CLOSE = '*' + '/';

function stripComments(css: string): string {
	let out = '';
	let at = 0;
	for (let open = css.indexOf(COMMENT_OPEN); open !== -1; open = css.indexOf(COMMENT_OPEN, at)) {
		out += css.slice(at, open);
		at = css.indexOf(COMMENT_CLOSE, open) + COMMENT_CLOSE.length;
	}
	return out + css.slice(at);
}

function topLevelBlockEnd(css: string, opener: string): number {
	let depth = 0;
	for (let i = css.indexOf(opener); i < css.length; i++) {
		if (css[i] === '{') depth += 1;
		if (css[i] === '}') {
			depth -= 1;
			if (depth === 0) return i;
		}
	}
	return -1;
}

describe('segment-composer.css', () => {
	const css = stripComments(readCss());

	it('has balanced braces', () => {
		const opens = css.split('{').length - 1;
		const closes = css.split('}').length - 1;
		expect(opens).toBe(closes);
	});

	it('keeps the global tail rules outside the .segment-composer nest', () => {
		const nestEnd = topLevelBlockEnd(css, '.segment-composer {');
		expect(nestEnd).toBeGreaterThan(0);
		const tail = css.slice(nestEnd + 1);
		expect(tail).toContain('.composer-count { font-size: 12px; }');
		expect(tail).toContain('.card.segment .segment-head');
		expect(css.slice(0, nestEnd)).toContain('.small-button:focus-visible');
		expect(tail).not.toContain('.small-button:focus-visible');
	});

	it.each(['.form-grid', '.field-label', '.swatches', '.swatch', '.swatch.active', '.swatch-current', '.clear-color'])(
		'no longer carries the unreferenced %s rule',
		(selector) => {
			const escaped = selector.replace(/\./g, '\\.');
			expect(css).not.toMatch(new RegExp(`(^|[\\s,}])${escaped}(?![\\w-])`));
		}
	);
});
