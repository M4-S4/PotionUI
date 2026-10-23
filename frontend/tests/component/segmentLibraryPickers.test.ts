// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { flushSync, tick } from 'svelte';
import type { SavedSegment, SegmentTemplate } from '$lib/types/segments';

const template = (id: string, name: string): SegmentTemplate => ({
	id,
	name,
	description: null,
	tags: [],
	segments: [
		{ type: 'content', content: `${name} subject`, chips: {}, enabled: true, name: 'Subject' },
		{ type: 'content', content: `${name} scene`, chips: {}, enabled: true, name: 'Scene' }
	]
});
const saved = (id: string, name: string, color: string | null): SavedSegment => ({
	id,
	name,
	category_id: 'cat-1',
	tags: ['studio'],
	type: 'content',
	content: `${name} content`,
	chips: {},
	enabled: true,
	effective_color: color
});

const listSegmentTemplates = vi.fn().mockResolvedValue({ success: true, data: { templates: [template('t1', 'Portrait'), template('t2', 'Product')] } });
const listSavedSegments = vi.fn().mockResolvedValue({ success: true, data: { segments: [saved('s1', 'Editorial subject', null), saved('s2', 'Golden hour', '#F59E0B')] } });
const listSegmentCategories = vi.fn().mockResolvedValue({ success: true, data: { categories: [{ id: 'cat-1', name: 'Subjects', description: '', color: '', user_id: '' }] } });

vi.mock('$lib/services/api/index', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$lib/services/api/index')>();
	return {
		...actual,
		api: {
			...actual.api,
			listSegmentTemplates: (...a: unknown[]) => listSegmentTemplates(...a),
			listSavedSegments: (...a: unknown[]) => listSavedSegments(...a),
			listSegmentCategories: (...a: unknown[]) => listSegmentCategories(...a)
		}
	};
});

const { default: SegmentListApplyModal } = await import('$lib/components/modals/SegmentListApplyModal.svelte');
const { default: SavedSegmentSelectionModal } = await import('$lib/components/modals/SavedSegmentSelectionModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

let component: { $destroy: () => void; $on: (name: string, fn: (e: CustomEvent) => void) => void } | undefined;

afterEach(() => {
	component?.$destroy();
	component = undefined;
	document.body.innerHTML = '';
});

async function settle() {
	for (let i = 0; i < 4; i++) {
		await Promise.resolve();
		await tick();
	}
}

function open(Component: unknown, props: Record<string, unknown>) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({ component: Component as never, target, props: { isOpen: true, ...props } }) as never;
	flushSync();
}

const named = (title: string) =>
	Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"], [role="alertdialog"]')).find(
		(el) => document.getElementById(el.getAttribute('aria-labelledby') ?? '')?.textContent === title
	);
const buttonStarting = (root: ParentNode, label: string) =>
	Array.from(root.querySelectorAll<HTMLButtonElement>('button')).find((b) => b.textContent?.trim().startsWith(label))!;

describe('Apply Segment Template', () => {
	it('uses the L width, defaults to Append and previews the selected template', async () => {
		open(SegmentListApplyModal, { kind: 'template', targetHasMeaningfulContent: false });
		await settle();
		const dialog = named('Apply Segment Template')!;
		expect(dialog.className).toContain('md:w-[1040px]');
		expect(dialog.querySelector<HTMLInputElement>('input[type="radio"][value="append"]')!.checked).toBe(true);
		expect(buttonStarting(dialog, 'Apply').disabled).toBe(true);
		const rows = Array.from(dialog.querySelectorAll<HTMLElement>('[data-library-row]'));
		rows[1].click();
		await settle();
		const preview = dialog.querySelector('aside[aria-label="Preview"]')!;
		expect(preview.querySelector('h3')!.textContent).toBe('Product');
		expect(preview.textContent).toContain('2 · Scene');
		expect(buttonStarting(dialog, 'Apply').disabled).toBe(false);
	});

	it('asks before replacing meaningful segments and applies on Replace', async () => {
		open(SegmentListApplyModal, { kind: 'template', targetHasMeaningfulContent: true });
		const applied: unknown[] = [];
		component!.$on('apply', (event) => applied.push(event.detail));
		await settle();
		const dialog = named('Apply Segment Template')!;
		dialog.querySelectorAll<HTMLElement>('[data-library-row]')[0].click();
		dialog.querySelector<HTMLInputElement>('input[type="radio"][value="replace"]')!.click();
		await settle();
		buttonStarting(dialog, 'Apply').click();
		await settle();
		expect(applied).toEqual([]);
		const confirm = named('Replace current segments?')!;
		expect(confirm.getAttribute('role')).toBe('alertdialog');
		buttonStarting(confirm, 'Replace').click();
		await settle();
		expect(applied).toHaveLength(1);
		expect(applied[0]).toMatchObject({ mode: 'replace', item: { id: 't1' } });
	});

	it.each([
		['Cancel', (confirm: HTMLElement) => buttonStarting(confirm, 'Cancel').click()],
		['Escape', () => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))]
	])('applies nothing and keeps the picker open when %s dismisses the replace question', async (_name, dismiss) => {
		open(SegmentListApplyModal, { kind: 'template', targetHasMeaningfulContent: true });
		const applied: unknown[] = [];
		component!.$on('apply', (event) => applied.push(event.detail));
		await settle();
		const dialog = named('Apply Segment Template')!;
		dialog.querySelectorAll<HTMLElement>('[data-library-row]')[0].click();
		dialog.querySelector<HTMLInputElement>('input[type="radio"][value="replace"]')!.click();
		await settle();
		buttonStarting(dialog, 'Apply').click();
		await settle();
		dismiss(named('Replace current segments?')!);
		await settle();
		expect(named('Replace current segments?')).toBeUndefined();
		expect(named('Apply Segment Template')).toBeDefined();
		expect(applied).toEqual([]);
	});
});

describe('saved Segment picker', () => {
	it('selects on click, inserts from the primary, and names the action after the flow', async () => {
		open(SavedSegmentSelectionModal, { title: 'Insert saved Segment' });
		const picked: string[] = [];
		component!.$on('select', (event) => picked.push(event.detail.savedSegment.id));
		await settle();
		const dialog = named('Insert saved Segment')!;
		const rows = Array.from(dialog.querySelectorAll<HTMLElement>('[data-saved-segment]'));
		expect(rows[0].getAttribute('aria-selected')).toBe('true');
		rows[1].click();
		await settle();
		expect(picked).toEqual([]);
		expect(rows[1].getAttribute('aria-selected')).toBe('true');
		expect(dialog.querySelector('aside[aria-label="Preview"]')!.textContent).toContain('Golden hour content');
		buttonStarting(dialog, 'Insert').click();
		await settle();
		expect(picked).toEqual(['s2']);
	});

	it('draws a neutral dot for a segment without a colour, never a fallback hex', async () => {
		open(SavedSegmentSelectionModal, {});
		await settle();
		const dots = Array.from(document.querySelectorAll<HTMLElement>('[data-testid="saved-segment-dot"]'));
		expect(dots[0].getAttribute('style')).toBeNull();
		expect(dots[0].className).toContain('bg-line-strong');
		expect(dots[1].style.backgroundColor).toBe('rgb(245, 158, 11)');
	});

	it('says nothing matches and clears the search', async () => {
		open(SavedSegmentSelectionModal, {});
		await settle();
		const search = document.querySelector<HTMLInputElement>('input[aria-label="Search saved Segments"]')!;
		search.value = 'zzzz';
		search.dispatchEvent(new Event('input', { bubbles: true }));
		await settle();
		const empty = document.querySelector('[data-testid="saved-segment-empty"]')!;
		expect(empty.textContent).toContain('zzzz');
		buttonStarting(empty, 'Clear search').click();
		await settle();
		expect(document.querySelectorAll('[data-saved-segment]')).toHaveLength(2);
	});
});
