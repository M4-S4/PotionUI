// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { get } from 'svelte/store';

vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$lib/media/editors/MediaEditors.svelte', async () => ({
	default: (await import('./fixtures/MediaEditorsStub.svelte')).default
}));

const { default: HistoryEditImageModal } = await import('../../src/routes/history/components/HistoryEditImageModal.svelte');
const { libraryModalRequest, closeLibraryResultModal } = await import('$lib/library/landing');

const context = {
	scope: 'history',
	items: [{ id: 'g1:0', kind: 'image', url: '/a.png', filename: 'a.png', generationId: 'g1', paramIndex: 2 }],
	kinds: new Set(['image']),
	collectionId: null,
	generations: [],
	generationIds: ['g1'],
	files: []
} as never;

let instance: Record<string, unknown> | undefined;
let target: HTMLElement | undefined;

beforeEach(() => {
	closeLibraryResultModal();
	target = document.createElement('div');
	document.body.appendChild(target);
});

afterEach(() => {
	if (instance) unmount(instance);
	target?.remove();
	instance = undefined;
	document.body.innerHTML = '';
});

describe('the History Edit image modal', () => {
	it('opens the Library modal after the editor has closed, not while it is still open', async () => {
		const onClose = vi.fn();
		instance = mount(HistoryEditImageModal, { target: target!, props: { context, onClose, onDone: vi.fn() } });
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-result]')!.click();
		await Promise.resolve();
		expect(get(libraryModalRequest)).toBeNull();

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		expect(onClose).toHaveBeenCalledTimes(1);
		await new Promise((resolve) => setTimeout(resolve, 5));

		expect(get(libraryModalRequest)?.item.id).toBe('looked');
		expect(get(libraryModalRequest)?.source).toEqual({ generationId: 'g1', fileIndex: 2 });
	});

	it('opens nothing when the editor is closed without a result', async () => {
		instance = mount(HistoryEditImageModal, { target: target!, props: { context, onClose: vi.fn(), onDone: vi.fn() } });
		flushSync();

		document.body.querySelector<HTMLButtonElement>('[data-stub-close]')!.click();
		await new Promise((resolve) => setTimeout(resolve, 5));

		expect(get(libraryModalRequest)).toBeNull();
	});
});
