import { beforeEach, describe, expect, it, vi } from 'vitest';

const goto = vi.fn();
const show = vi.fn();
vi.mock('$app/navigation', () => ({ goto: (...args: unknown[]) => goto(...args) }));
vi.mock('$lib/stores/toast', () => ({ toasts: { show: (...args: unknown[]) => show(...args) } }));

import { notifySavedToLibrary, savedToLibraryMessage } from './savedToLibrary';

const item = (name: string) => ({ id: name, filename: `${name}-uuid.mp4`, original_filename: name, media_type: 'video', url: '/u' });

beforeEach(() => {
	goto.mockReset();
	show.mockReset();
});

describe('savedToLibraryMessage', () => {
	it('names a single result', () => {
		expect(savedToLibraryMessage([item('clip-speed.mp4')])).toBe('clip-speed.mp4 is in your Library.');
	});

	it('counts several results', () => {
		expect(savedToLibraryMessage([item('a'), item('b'), item('c')])).toBe('3 items are in your Library.');
	});
});

describe('notifySavedToLibrary', () => {
	it('shows a success toast whose action opens the Library', () => {
		notifySavedToLibrary([item('clip.mp4')]);

		expect(show).toHaveBeenCalledTimes(1);
		const [type, message, options] = show.mock.calls[0];
		expect(type).toBe('success');
		expect(message).toContain('clip.mp4');
		expect(options.action.label).toBe('Open Library');
		options.action.onClick();
		expect(goto).toHaveBeenCalledWith('/library');
	});

	it('says nothing when nothing was saved', () => {
		notifySavedToLibrary([]);
		expect(show).not.toHaveBeenCalled();
	});
});
