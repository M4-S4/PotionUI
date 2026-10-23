import { describe, it, expect } from 'vitest';
import { TERMINAL_TYPES, generationIdOf } from './wire';

describe('TERMINAL_TYPES', () => {
	it('is exactly the three terminal frame types', () => {
		expect([...TERMINAL_TYPES].sort()).toEqual(['generation_cancelled', 'generation_complete', 'generation_error']);
	});
});

describe('generationIdOf', () => {
	it('reads the top-level id for progress frames', () => {
		expect(generationIdOf({ type: 'generation_status', generation_id: 'g1', data: { id: 'x' } })).toBe('g1');
	});

	it('reads data.id on complete and cancelled frames', () => {
		expect(generationIdOf({ type: 'generation_complete', data: { id: 'g2' } })).toBe('g2');
		expect(generationIdOf({ type: 'generation_cancelled', data: { id: 'g3' } })).toBe('g3');
	});

	it('reads data.generation_id on an error frame that has no data.id', () => {
		expect(generationIdOf({ type: 'generation_error', data: { generation_id: 'g4' } })).toBe('g4');
	});

	it('falls back to the top-level id on a terminal frame without data ids', () => {
		expect(generationIdOf({ type: 'generation_error', generation_id: 'g5' })).toBe('g5');
		expect(generationIdOf({ type: 'generation_complete', generation_id: 'g6', data: { id: '' } })).toBe('g6');
	});

	it('prefers data.id over data.generation_id', () => {
		expect(generationIdOf({ type: 'generation_complete', data: { id: 'a', generation_id: 'b' } })).toBe('a');
	});

	it('returns undefined when no id is present', () => {
		expect(generationIdOf({ type: 'generation_complete', data: {} })).toBeUndefined();
	});
});
