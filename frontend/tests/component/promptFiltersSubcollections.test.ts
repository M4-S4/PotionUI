import { describe, it, expect, vi, afterEach } from 'vitest';

vi.mock('$lib/services/api', () => ({
	api: { listPromptTags: vi.fn().mockResolvedValue({ success: true, data: { tags: [] } }) }
}));

const { default: PromptFiltersPopover } = await import('../../src/lib/prompts/PromptFiltersPopover.svelte');
const { DEFAULT_PROMPT_FILTERS } = await import('../../src/lib/prompts/promptFilters');
const { mountComponent } = await import('./helpers/mount');
const { settle } = await import('./helpers/settle');

const collections = [
	{ id: 'photo', parent_id: null, name: 'Photography', item_count: 28 },
	{ id: 'portraits', parent_id: 'photo', name: 'Portraits', item_count: 12 }
];

const includeSwitch = () =>
	document.querySelector<HTMLInputElement>('[data-testid="include-subcollections"] input[role="switch"]');

function mountPopover(overrides: Record<string, unknown> = {}) {
	const props = {
		filters: { ...DEFAULT_PROMPT_FILTERS },
		onChange: vi.fn(),
		onOpenModelPicker: vi.fn(),
		onClose: vi.fn(),
		onClearAll: vi.fn(),
		collections,
		activeId: 'photo',
		directOnly: false,
		onDirectOnlyChange: vi.fn(),
		...overrides
	};
	mountComponent(PromptFiltersPopover as never, props);
	return props;
}

afterEach(() => {
	document.body.innerHTML = '';
});

describe('prompt filters: include subcollections', () => {
	it('shows the row only for a collection with subcollections on a page that handles it', () => {
		mountPopover({ activeId: 'portraits' });
		expect(includeSwitch()).toBeNull();
		document.body.innerHTML = '';
		mountPopover({ onDirectOnlyChange: undefined });
		expect(includeSwitch()).toBeNull();
		document.body.innerHTML = '';
		mountPopover();
		expect(includeSwitch()?.checked).toBe(true);
	});

	it('reports direct only when switched off', async () => {
		const props = mountPopover();
		includeSwitch()!.click();
		await settle();
		expect(props.onDirectOnlyChange).toHaveBeenCalledWith(true);
	});

	it('hands Clear all to the page so it can reset direct only with the other filters', async () => {
		const props = mountPopover({ directOnly: true });
		Array.from(document.querySelectorAll<HTMLButtonElement>('button'))
			.find((b) => b.textContent?.trim() === 'Clear all')!
			.click();
		await settle();
		expect(props.onClearAll).toHaveBeenCalledTimes(1);
		expect(props.onChange).not.toHaveBeenCalled();
	});
});
