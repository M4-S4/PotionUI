// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync, mount, tick, unmount } from 'svelte';
import { PRESET_COLORS, type Segment } from '$lib/types/segments';

const listSegmentCategories = vi.fn().mockResolvedValue({
	success: true,
	data: { categories: [{ id: 'cat-1', name: 'General', description: '', color: '', user_id: '' }] }
});
const createSavedSegment = vi.fn().mockResolvedValue({ success: true, data: {} });
const createSegmentTemplate = vi.fn().mockResolvedValue({ success: true, data: { id: 't1' } });

vi.mock('$lib/services/api', () => ({
	api: {
		listSegmentCategories: (...a: unknown[]) => listSegmentCategories(...a),
		createSavedSegment: (...a: unknown[]) => createSavedSegment(...a),
		createSegmentTemplate: (...a: unknown[]) => createSegmentTemplate(...a)
	}
}));

const { default: SaveSegmentModal } = await import('$lib/components/modals/SaveSegmentModal.svelte');
const { default: SegmentTemplateSaveModal } = await import('$lib/components/modals/SegmentTemplateSaveModal.svelte');
const { default: PromptSegmentDetailsModal } = await import('$lib/components/PromptSegmentDetailsModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

const segment: Segment = { id: 'seg-1', name: '', content: 'a cat', chips: {}, enabled: true };
let instance: ReturnType<typeof mount> | undefined;

afterEach(() => {
	if (instance) unmount(instance);
	instance = undefined;
	listSegmentCategories.mockClear();
	createSavedSegment.mockClear();
	createSegmentTemplate.mockClear();
	document.body.innerHTML = '';
});

async function settle() {
	await Promise.resolve();
	await tick();
	await tick();
}

function pressEnter(target: Element) {
	target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
}

describe('required names', () => {
	it('keeps Save as Segment open and says what is missing until a name is typed', async () => {
		const target = document.createElement('div');
		document.body.appendChild(target);
		createClassComponent({ component: SaveSegmentModal as never, target, props: { isOpen: true, segment } });
		flushSync();
		await settle();
		const name = document.querySelector<HTMLInputElement>('#saved-segment-name')!;
		pressEnter(name);
		await settle();
		expect(createSavedSegment).not.toHaveBeenCalled();
		expect(name.getAttribute('aria-invalid')).toBe('true');
		expect(document.querySelector('[data-testid="field-problem"]')!.textContent).toContain('Give it a name to save it.');
		name.value = 'Studio style';
		name.dispatchEvent(new Event('input', { bubbles: true }));
		await settle();
		expect(document.querySelector('[data-testid="field-problem"]')).toBeNull();
		pressEnter(name);
		await settle();
		expect(createSavedSegment).toHaveBeenCalledTimes(1);
		expect(createSavedSegment.mock.calls[0][0]).toMatchObject({ name: 'Studio style', category_id: 'cat-1' });
	});

	it('saves the colour picked from the swatches', async () => {
		const target = document.createElement('div');
		document.body.appendChild(target);
		createClassComponent({ component: SaveSegmentModal as never, target, props: { isOpen: true, segment: { ...segment, name: 'Studio' } } });
		flushSync();
		await settle();
		document.querySelector<HTMLButtonElement>(`button[aria-label="${PRESET_COLORS[2].name}"]`)!.click();
		await settle();
		expect(document.querySelector(`button[aria-label="${PRESET_COLORS[2].name}"]`)!.getAttribute('aria-pressed')).toBe('true');
		pressEnter(document.querySelector('#saved-segment-name')!);
		await settle();
		expect(createSavedSegment.mock.calls[0][0]).toMatchObject({ color: PRESET_COLORS[2].value });
	});

	it('keeps Save as Segment Template open until it has a name', async () => {
		const onSaved = vi.fn();
		instance = mount(SegmentTemplateSaveModal, { target: document.body, props: { isOpen: true, segments: [segment], onClose: () => {}, onSaved } });
		flushSync();
		await settle();
		const name = document.querySelector<HTMLInputElement>('#segment-template-name')!;
		pressEnter(name);
		await settle();
		expect(createSegmentTemplate).not.toHaveBeenCalled();
		expect(name.getAttribute('aria-invalid')).toBe('true');
		name.value = 'Portrait';
		name.dispatchEvent(new Event('input', { bubbles: true }));
		pressEnter(name);
		await settle();
		expect(createSegmentTemplate).toHaveBeenCalledTimes(1);
		expect(onSaved).toHaveBeenCalledTimes(1);
	});
});

describe('segment details form', () => {
	it('labels every field in readable text and names the segment in the header', async () => {
		instance = mount(PromptSegmentDetailsModal, {
			target: document.body,
			props: { isOpen: true, segment: { ...segment, name: 'Subject' }, onClose: () => {}, onSave: () => {} }
		});
		flushSync();
		await settle();
		const labels = Array.from(document.querySelectorAll('label[for^="segment-details-"]'));
		expect(labels.map((l) => l.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
			'Name (optional)',
			'Prefix',
			'Suffix',
			'Description (optional)'
		]);
		labels.forEach((l) => expect(l.className).toContain('text-sm'));
		expect(document.querySelector('[data-segment-modal-context]')!.textContent).toBe('Subject');
	});
});
