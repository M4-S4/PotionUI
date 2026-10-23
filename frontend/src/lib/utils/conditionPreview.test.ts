import { describe, expect, it } from 'vitest';
import { previewConditions } from './conditionPreview';
import type { VariablesMap } from './variableDefs';

const TEMPLATE = 'portrait, <<if $age > 65>>grey hair<<else if $age < 25>>youthful<<else>>auburn hair<<end>>, soft light';

const per = (over = {}): VariablesMap => ({
	age: { type: 'number', mode: 'per-image', min: 18, max: 80, step: 1, value: 18, ...over } as never
});

describe('previewConditions', () => {
	it('chooses the branch when every read variable is settled', () => {
		const result = previewConditions(TEMPLATE, per({ mode: 'pin', value: 70 }));
		expect(result.text).toBe('portrait, grey hair, soft light');
		expect(result.rows).toEqual([]);
		expect(result.notes).toEqual([]);
	});

	it('takes the otherwise branch when nothing matches', () => {
		expect(previewConditions(TEMPLATE, per({ mode: 'pin', value: 40 })).text).toBe('portrait, auburn hair, soft light');
	});

	it('shows every branch and a note while a read variable varies', () => {
		const result = previewConditions(TEMPLATE, per());
		expect(result.text).toBe('portrait, grey hair / youthful / auburn hair, soft light');
		expect(result.notes).toEqual([{ name: 'age', phrase: 're-rolls per image — the condition follows it' }]);
	});

	it('words the note for a shuffle variable', () => {
		const result = previewConditions(TEMPLATE, per({ mode: 'shuffle' }));
		expect(result.notes[0].phrase).toBe('shuffles each generation — the condition follows it');
	});

	it('builds one row per image for a per-image variable, with the branch each takes', () => {
		const result = previewConditions(TEMPLATE, per());
		expect(result.rows).toHaveLength(4);
		for (const row of result.rows) {
			const age = Number(row.rolled[0].value);
			const expected = age > 65 ? 'IF' : age < 25 ? 'ELSE IF' : 'OTHERWISE';
			expect(row.branch).toBe(expected);
			const hit = row.pieces.find((p) => p.hit)!;
			expect(hit.text).toBe(age > 65 ? 'grey hair' : age < 25 ? 'youthful' : 'auburn hair');
		}
		expect(result.rows.map((r) => r.label)).toEqual(['Image 1', 'Image 2', 'Image 3', 'Image 4']);
	});

	it('is deterministic', () => {
		expect(previewConditions(TEMPLATE, per())).toEqual(previewConditions(TEMPLATE, per()));
	});

	it('collapses a shuffle variable to a single row', () => {
		const result = previewConditions(TEMPLATE, per({ mode: 'shuffle' }));
		expect(result.rows).toHaveLength(1);
		expect(result.rows[0].label).toBe('All 4');
	});

	it('substitutes the rolled value into the row text', () => {
		const result = previewConditions(`${TEMPLATE}, ${'$'}{age} years old`, per());
		for (const row of result.rows) {
			const text = row.pieces.map((p) => p.text).join('');
			expect(text.endsWith(`, ${row.rolled[0].value} years old`)).toBe(true);
		}
	});

	it('leaves text without conditions alone', () => {
		expect(previewConditions('plain ${age}', per())).toEqual({ text: 'plain ${age}', notes: [], rows: [] });
	});
});
