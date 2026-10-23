import { describe, expect, it } from 'vitest';
import {
	MAX_HEAD_CHARS,
	MAX_HEAD_SCAN_CHARS,
	MAX_SCAN_CHARS,
	parseTemplate,
	renderConditions
} from './promptConditions';

const fill = (unit: string, size = MAX_SCAN_CHARS) => unit.repeat(Math.ceil(size / unit.length)).slice(0, size);

const floods: Record<string, string> = {
	'if openers': fill('<<if '),
	'if backslash openers': fill('<<if \\'),
	'else if openers': fill('<<else if '),
	'quote and escape after if': fill('<<if  "\\"'),
	'unclosed quotes': fill('<<if $a = "x '),
	'entity heads': fill('<<if $a &gt; '),
	'heads without a closer': fill('<<if $a > 1 '),
	'valid tags with bad heads': fill('<<if $a>1>>x<<end>>'),
	'480 KB of if-end pairs': '<<if<<end>> '.repeat(40000),
	'480 KB of backslash openers': fill('<<if \\', 480_000)
};

describe('tag scanning cost', () => {
	it.each(Object.entries(floods))('%s parses in well under the scan budget', (_name, text) => {
		parseTemplate(text);
		const start = performance.now();
		parseTemplate(text);
		expect(performance.now() - start).toBeLessThan(60 * Math.max(1, Math.floor(text.length / (8 * MAX_SCAN_CHARS))));
	});

	it('reads heads with one allowance per scan', () => {
		expect(MAX_HEAD_SCAN_CHARS).toBe(4 * MAX_SCAN_CHARS);
		const spent = MAX_HEAD_CHARS + 2;
		const stillRead = Math.floor(MAX_HEAD_SCAN_CHARS / spent);
		const tail = '"' + 'y'.repeat(spent + 100) + ' <<if $a > 1>>A<<else>>B<<end>>';
		expect(renderConditions('<<if \\'.repeat(stillRead) + tail, { a: '0' }).text.endsWith(' B')).toBe(true);
		expect(renderConditions('<<if \\'.repeat(stillRead + 1) + tail, { a: '0' }).text.endsWith('>>AB')).toBe(true);
	});
});
