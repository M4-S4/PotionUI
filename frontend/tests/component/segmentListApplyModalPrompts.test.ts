import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Prompt } from '$lib/types/segments';
import { settle } from './helpers/settle';

vi.mock('$lib/services/api/index', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$lib/services/api/index')>();
	return {
		...actual,
		api: {
			...actual.api,
			listPrompts: vi.fn(),
			getModels: vi.fn(),
			listPromptTags: vi.fn()
		}
	};
});

const { api } = await import('$lib/services/api/index');
const { default: SegmentListApplyModal } = await import('$lib/components/modals/SegmentListApplyModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

function prompt(overrides: Partial<Prompt> = {}): Prompt {
	return {
		id: 'prompt-1',
		name: 'Dancing girl factory',
		display_name: 'Dancing girl factory',
		segments: [],
		flattened_text: 'She eases into ${dance} keeping eye contact with the lens.',
		usage_hint: 'positive',
		model_name: 'MiniMax-H3',
		usage_count: 12,
		last_used_at: '2026-09-22T00:00:00.000Z',
		variables: {
			dance: { type: 'choice', options: ['breaking', 'waltz'], mode: 'shuffle', pinnedIndex: null }
		},
		...overrides
	};
}

const rooftop = prompt({ id: 'prompt-2', name: 'Afrobeats waist wind', flattened_text: 'a dancer on a rooftop at dusk', variables: null });

function mountModal() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: SegmentListApplyModal as never,
		target,
		props: { isOpen: true, kind: 'prompt', targetHasMeaningfulContent: false }
	});
	return {
		component,
		destroy: () => {
			component.$destroy();
			target.remove();
		}
	};
}

async function waitDebounce() {
	await vi.advanceTimersByTimeAsync(300);
	await settle();
}

function dialog(): HTMLElement {
	const el = document.body.querySelector('[role="dialog"][aria-label="Apply Prompt"]');
	if (!el) throw new Error('Apply Prompt dialog not mounted');
	return el as HTMLElement;
}

function rows(): HTMLElement[] {
	return Array.from(dialog().querySelectorAll('[data-library-row]'));
}

function searchInput(): HTMLInputElement {
	return dialog().querySelector('input[type="search"]') as HTMLInputElement;
}

function findButton(root: HTMLElement, predicate: (text: string) => boolean): HTMLButtonElement | undefined {
	return Array.from(root.querySelectorAll('button')).find((button) => predicate((button.textContent || '').trim())) as
		| HTMLButtonElement
		| undefined;
}

function key(el: Element, keyName: string) {
	el.dispatchEvent(new KeyboardEvent('keydown', { key: keyName, bubbles: true, cancelable: true }));
}

let mounted: ReturnType<typeof mountModal> | undefined;

beforeEach(() => {
	vi.useFakeTimers();
	vi.mocked(api.listPrompts).mockResolvedValue({
		success: true,
		data: { items: [prompt(), rooftop], total: 2, limit: 48, offset: 0 }
	} as never);
	vi.mocked(api.getModels).mockResolvedValue({ success: true, data: { models: [] } } as never);
	vi.mocked(api.listPromptTags).mockResolvedValue({ success: true, data: { tags: [] } } as never);
});

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	document.body.innerHTML = '';
	vi.clearAllMocks();
	vi.useRealTimers();
});

describe('SegmentListApplyModal prompt library', () => {
	it('renders the prompt rows the api returns, with the ${var} usage in signal mono', async () => {
		mounted = mountModal();
		await settle();
		expect(rows()).toHaveLength(2);
		expect(rows()[0].textContent).toContain('Dancing girl factory');
		expect(rows()[1].textContent).toContain('Afrobeats waist wind');
		const varSpan = Array.from(rows()[0].querySelectorAll('span')).find((el) => el.textContent === '${dance}');
		expect(varSpan?.className).toContain('text-signal');
		expect(rows()[0].textContent).toContain('used 12×');
		expect(vi.mocked(api.listPrompts).mock.calls[0][0]).toMatchObject({ limit: 48, offset: 0, nsfw: 'exclude', sort_by: 'last_used_at' });
		expect(vi.mocked(api.listPrompts).mock.calls[0][0]?.semantic).toBeUndefined();
	});

	it('asks for semantic matches with a typed query and shows the semantic hint', async () => {
		mounted = mountModal();
		await settle();
		vi.mocked(api.listPrompts).mockResolvedValue({
			success: true,
			data: { items: [prompt(), rooftop], total: 2, limit: 48, offset: 0, semantic_hits: 1 }
		} as never);
		const input = searchInput();
		input.value = 'dancer';
		input.dispatchEvent(new Event('input', { bubbles: true }));
		await waitDebounce();
		expect(vi.mocked(api.listPrompts).mock.calls.at(-1)?.[0]).toMatchObject({ q: 'dancer', offset: 0, semantic: true });
		expect(dialog().textContent).toContain('semantic · 1 hits');
		expect(rows()[0].textContent).toContain('Dancing girl factory');
		expect(rows()[1].textContent).toContain('Afrobeats waist wind');
	});

	it('counts a changed filter on the Filters button and lists it as a chip', async () => {
		mounted = mountModal();
		await settle();
		const filtersButton = findButton(dialog(), (text) => text.startsWith('Filters'));
		expect(filtersButton).toBeTruthy();
		filtersButton!.click();
		await settle();
		const popover = dialog().querySelector('[role="dialog"][aria-label="Filters"]') as HTMLElement;
		expect(popover).toBeTruthy();
		findButton(popover, (text) => text === 'Never')!.click();
		await settle();
		expect(filtersButton!.textContent?.replace(/\s+/g, ' ').trim()).toBe('Filters 1');
		expect(dialog().textContent).toContain('never used');
		expect(dialog().textContent).toContain('2 of 2');
		await waitDebounce();
		expect(vi.mocked(api.listPrompts).mock.calls.at(-1)?.[0]).toMatchObject({ used: 'never' });
	});

	it('applies the keyboard-selected row with the chosen mode on Enter', async () => {
		mounted = mountModal();
		await settle();
		const onApply = vi.fn();
		mounted.component.$on('apply', (event: CustomEvent) => onApply(event.detail));
		const appendRadio = dialog().querySelector('input[type="radio"][value="append"]') as HTMLInputElement;
		appendRadio.click();
		await settle();
		key(searchInput(), 'ArrowDown');
		await settle();
		expect(rows()[0].getAttribute('aria-selected')).toBe('true');
		key(searchInput(), 'ArrowDown');
		await settle();
		expect(rows()[1].getAttribute('aria-selected')).toBe('true');
		key(searchInput(), 'Enter');
		await settle();
		expect(onApply).toHaveBeenCalledTimes(1);
		expect(onApply.mock.calls[0][0].item.id).toBe('prompt-2');
		expect(onApply.mock.calls[0][0].mode).toBe('append');
	});

	it('requests the next offset from Show more and appends the page', async () => {
		vi.mocked(api.listPrompts).mockResolvedValue({
			success: true,
			data: { items: [prompt(), rooftop], total: 5, limit: 48, offset: 0 }
		} as never);
		mounted = mountModal();
		await settle();
		const showMore = findButton(dialog(), (text) => text === 'Show 3 more');
		expect(showMore).toBeTruthy();
		vi.mocked(api.listPrompts).mockResolvedValue({
			success: true,
			data: { items: [prompt({ id: 'prompt-3', name: 'Third' })], total: 5, limit: 48, offset: 2 }
		} as never);
		showMore!.click();
		await settle();
		expect(vi.mocked(api.listPrompts).mock.calls.at(-1)?.[0]).toMatchObject({ offset: 2, limit: 48 });
		expect(rows()).toHaveLength(3);
		expect(rows()[2].textContent).toContain('Third');
	});
});

describe('SegmentListApplyModal filters popover dismissal', () => {
	const filtersTrigger = () => findButton(dialog(), (text) => text.startsWith('Filters'))!;
	const popover = () => dialog().querySelector<HTMLElement>('[role="dialog"]');
	const pickerOpen = () => document.body.textContent?.includes('Filter prompts by model') ?? false;

	function pointerdown(el: Element) {
		el.dispatchEvent(new Event('pointerdown', { bubbles: true, cancelable: true }));
	}

	function escape() {
		window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
	}

	async function openPopover() {
		mounted = mountModal();
		await settle();
		pointerdown(filtersTrigger());
		filtersTrigger().click();
		await settle();
		expect(popover()).not.toBeNull();
	}

	it('closes on Escape', async () => {
		await openPopover();
		escape();
		await settle();
		expect(popover()).toBeNull();
	});

	it('ends closed when the trigger is pressed while open', async () => {
		await openPopover();
		pointerdown(filtersTrigger());
		filtersTrigger().click();
		await settle();
		expect(popover()).toBeNull();
	});

	it('closes on an outside press', async () => {
		await openPopover();
		pointerdown(searchInput());
		await settle();
		expect(popover()).toBeNull();
	});

	it('keeps the popover and dialog when Escape closes the nested model picker', async () => {
		await openPopover();
		findButton(popover()!, (text) => text.includes('Any'))!.click();
		await settle();
		expect(pickerOpen()).toBe(true);

		escape();
		await settle();

		expect(pickerOpen()).toBe(false);
		expect(popover()).not.toBeNull();
		expect(dialog()).toBeTruthy();
	});

	it('keeps the popover open on a press inside the model picker', async () => {
		await openPopover();
		findButton(popover()!, (text) => text.includes('Any'))!.click();
		await settle();
		expect(pickerOpen()).toBe(true);

		const heading = Array.from(document.body.querySelectorAll('*')).find(
			(el) => el.children.length === 0 && el.textContent === 'Filter prompts by model'
		)!;
		pointerdown(heading);
		await settle();

		expect(popover()).not.toBeNull();
		expect(pickerOpen()).toBe(true);
	});
});
