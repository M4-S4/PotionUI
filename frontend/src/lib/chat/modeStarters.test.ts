import { describe, expect, it } from 'vitest';
import { startersForMode } from './modeStarters';

const BUILTIN = ['generation', 'history', 'prompts', 'models', 'phrasebook'];

describe('startersForMode', () => {
	it('gives every builtin mode three complete starters', () => {
		for (const mode of BUILTIN) {
			const starters = startersForMode(mode);
			expect(starters).toHaveLength(3);
			for (const starter of starters) {
				expect(starter.title && starter.sub && starter.prompt && starter.icon).toBeTruthy();
			}
		}
	});

	it('keeps the image-oriented starters for the generation mode only', () => {
		expect(startersForMode('generation').map((s) => s.title)).toContain('Refine this prompt');
		for (const mode of BUILTIN.filter((id) => id !== 'generation')) {
			const text = startersForMode(mode)
				.map((s) => `${s.title} ${s.sub} ${s.prompt}`)
				.join(' ');
			expect(text).not.toContain('current image');
			expect(text).not.toContain('art directions');
		}
	});

	it('uses distinct titles per mode and a neutral set for unknown modes', () => {
		const titles = BUILTIN.flatMap((mode) => startersForMode(mode).map((s) => s.title));
		expect(new Set(titles).size).toBe(titles.length);
		expect(startersForMode('some-plugin-mode')).toHaveLength(3);
		expect(startersForMode(null)).toEqual(startersForMode('some-plugin-mode'));
		expect(startersForMode('some-plugin-mode')).not.toEqual(startersForMode('generation'));
	});
});
