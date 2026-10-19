import { describe, expect, it } from 'vitest';
import { coerceProviderOptionText, contextWindowReadout, migrateLegacyContextWindow } from './llmProviderOptions';

describe('coerceProviderOptionText', () => {
	it.each(['5m', '1h', '30s', '1.5h', '10ms'])('keeps the duration %s verbatim', (input) => {
		expect(coerceProviderOptionText(input)).toBe(input);
	});

	it.each([
		['0', 0],
		['-1', -1],
		['300', 300],
		['0.5', 0.5],
		['-0.25', -0.25]
	])('coerces the numeric literal %s', (input, expected) => {
		expect(coerceProviderOptionText(input as string)).toBe(expected);
	});

	it('coerces zero to a number rather than falling back to the string', () => {
		expect(coerceProviderOptionText('0')).not.toBe('0');
	});

	it('tolerates surrounding whitespace on a numeric literal', () => {
		expect(coerceProviderOptionText('  42  ')).toBe(42);
	});

	it.each(['', '   '])('returns blank input unchanged (%p)', (input) => {
		expect(coerceProviderOptionText(input)).toBe(input);
	});

	it.each(['indefinite', 'Infinity', '-Infinity', 'NaN', 'm5'])(
		'leaves the non-numeric value %s as text',
		(input) => {
			expect(coerceProviderOptionText(input)).toBe(input);
		}
	);
});

describe('migrateLegacyContextWindow', () => {
	it('moves num_ctx into context_window and drops num_ctx', () => {
		expect(migrateLegacyContextWindow({ num_ctx: 8192, top_k: 40 })).toEqual({ context_window: 8192, top_k: 40 });
	});

	it('keeps an existing context_window and still drops num_ctx', () => {
		expect(migrateLegacyContextWindow({ num_ctx: 8192, context_window: 16384 })).toEqual({ context_window: 16384 });
	});

	it('returns options without num_ctx untouched', () => {
		const options = { top_k: 40 };
		expect(migrateLegacyContextWindow(options)).toBe(options);
	});

	it('leaves the field blank for a non-numeric num_ctx', () => {
		expect(migrateLegacyContextWindow({ num_ctx: null })).toEqual({});
	});
});

describe('contextWindowReadout', () => {
	const base = { supported: true, detected: null, detected_kind: null, effective: null, discovered_at: null, error: null };
	const text = (r: ReturnType<typeof contextWindowReadout>) =>
		r?.parts.map((p) => (typeof p === 'string' ? p : p.n.toLocaleString())).join('');

	it('reports a runtime-detected window when the field is blank', () => {
		const r = contextWindowReadout({ ...base, detected: 32768, detected_kind: 'runtime', effective: 32768 }, false, null);
		expect(text(r)).toBe(`Using ${(32768).toLocaleString()} tokens, detected from the server`);
	});

	it('explains a trained window capped below what the model supports', () => {
		const r = contextWindowReadout({ ...base, detected: 262144, detected_kind: 'trained', effective: 32768 }, false, null);
		expect(text(r)).toBe(
			`Using ${(32768).toLocaleString()} tokens. The model supports ${(262144).toLocaleString()}; raise the window if your GPU has room`
		);
	});

	it('shows only a hint when the field is set', () => {
		const r = contextWindowReadout({ ...base, detected: 32768, detected_kind: 'runtime', effective: 4096 }, true, null);
		expect(text(r)).toBe(`Detected ${(32768).toLocaleString()} tokens`);
	});

	it('asks for a manual window when the provider cannot report one', () => {
		expect(text(contextWindowReadout({ ...base, supported: false }, false, null))).toBe(
			"This provider can't report its window — set it here"
		);
		expect(contextWindowReadout({ ...base, supported: false }, true, null)).toBeNull();
	});

	it('names the fallback after an error until a window is set', () => {
		const r = contextWindowReadout({ ...base, error: 'Server unreachable.' }, false, null);
		expect(text(r)).toBe(`Server unreachable. Using ${(8192).toLocaleString()} tokens until a window is set.`);
		expect(r?.tone).toBe('warning');
		expect(text(contextWindowReadout(null, true, 'Boom'))).toBe('Boom');
	});

	it('shows nothing before any result', () => {
		expect(contextWindowReadout(null, false, null)).toBeNull();
	});
});
