import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const css = readFileSync(new URL('./tokens.css', import.meta.url), 'utf8');
const tailwind = readFileSync(new URL('../../../tailwind.config.js', import.meta.url), 'utf8');

function block(opener: string): string {
	const start = css.indexOf(opener);
	return css.slice(start, css.indexOf('\n}\n', start));
}

function token(scope: string, name: string): string {
	const match = new RegExp(`--${name}:\\s*([0-9 ]+);`).exec(scope);
	if (!match) throw new Error(`--${name} is not declared`);
	return match[1].trim();
}

describe('the control fill token', () => {
	const dark = block(':root {');
	const light = block(":root[data-theme='light'] {");

	it('stays the surface-2 fill in the dark theme', () => {
		expect(token(dark, 'control-fill')).toBe(token(dark, 'surface-2'));
	});

	it('lifts to the surface-1 fill in the light theme so controls stand off the canvas', () => {
		expect(token(light, 'control-fill')).toBe(token(light, 'surface-1'));
		expect(token(light, 'control-fill')).not.toBe(token(light, 'surface-2'));
		expect(token(light, 'control-fill')).not.toBe(token(light, 'canvas'));
	});

	it('hovers to the surface-3 fill in dark and to a visibly different fill in light', () => {
		expect(token(dark, 'control-fill-hover')).toBe(token(dark, 'surface-3'));
		expect(token(dark, 'control-fill-hover')).not.toBe(token(dark, 'control-fill'));
		expect(token(light, 'control-fill-hover')).not.toBe(token(light, 'control-fill'));
		expect(token(light, 'control-fill-hover')).not.toBe(token(light, 'surface-3'));
	});

	it('is available as Tailwind colours', () => {
		expect(tailwind).toContain("'control-fill': t('control-fill')");
		expect(tailwind).toContain("'control-fill-hover': t('control-fill-hover')");
	});
});
