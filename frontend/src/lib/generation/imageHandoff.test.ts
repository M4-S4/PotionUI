import { describe, expect, it } from 'vitest';
import { createImageHandoff } from './imageHandoff';

const PREVIEW = 'data:image/png;base64,last-preview';
const FINAL = '/api/media/generations/gen-1/0.png';

function setup() {
	const shown: Array<string | null | undefined> = [];
	const pending: Array<() => void> = [];
	const handoff = createImageHandoff(
		(url) => shown.push(url),
		(_url, onSettled) => pending.push(onSettled)
	);
	return { shown, pending, handoff };
}

describe('createImageHandoff', () => {
	it('keeps the last preview on screen until the final image has loaded', () => {
		const { shown, pending, handoff } = setup();
		handoff(PREVIEW);
		handoff(FINAL);

		expect(shown).toEqual([PREVIEW]);

		pending[0]();

		expect(shown).toEqual([PREVIEW, FINAL]);
	});

	it('shows a first image immediately and swaps previews immediately', () => {
		const { shown, pending, handoff } = setup();
		handoff(PREVIEW);
		handoff('data:image/png;base64,next-preview');

		expect(shown).toEqual([PREVIEW, 'data:image/png;base64,next-preview']);
		expect(pending).toHaveLength(0);
	});

	it('clears immediately when there is nothing to show', () => {
		const { shown, handoff } = setup();
		handoff(PREVIEW);
		handoff(null);

		expect(shown).toEqual([PREVIEW, null]);
	});

	it('drops a stale final image that settles after a newer value', () => {
		const { shown, pending, handoff } = setup();
		handoff(PREVIEW);
		handoff(FINAL);
		handoff(null);
		pending[0]();

		expect(shown).toEqual([PREVIEW, null]);
	});

	it('does not hold when moving between saved images', () => {
		const { shown, pending, handoff } = setup();
		handoff(FINAL);
		handoff('/api/media/generations/gen-1/1.png');

		expect(shown).toEqual([FINAL, '/api/media/generations/gen-1/1.png']);
		expect(pending).toHaveLength(0);
	});
});
