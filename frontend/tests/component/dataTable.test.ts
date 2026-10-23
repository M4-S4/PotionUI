import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { mount, unmount, flushSync, createRawSnippet } from 'svelte';

const { default: DataTable } = await import('../../src/lib/components/table/DataTable.svelte');

import type { DataTableColumn } from '../../src/lib/components/table/column';

type Row = { id: string; name: string; email: string };

const ROWS: Row[] = [
	{ id: 'a', name: 'admin', email: 'admin@potionui.local' },
	{ id: 'b', name: 'someuser', email: 'someuser@potionui.local' }
];

const columns: DataTableColumn<any>[] = [
	{ key: 'name', label: 'Username', width: '1fr', sortable: true, accessor: (row: Row) => row.name },
	{ key: 'email', label: 'Email', width: '200px', accessor: (row: Row) => row.email },
	{ key: 'internal', label: 'Internal', width: '80px', priority: 2 as const, accessor: () => 'secret' }
];

let width = 0;
const observers: { targets: Set<Element>; callback: (entries: unknown[]) => void }[] = [];
globalThis.ResizeObserver = class {
	targets = new Set<Element>();
	constructor(public callback: (entries: unknown[]) => void) {
		observers.push(this);
	}
	observe(el: Element) {
		this.targets.add(el);
	}
	unobserve(el: Element) {
		this.targets.delete(el);
	}
	disconnect() {
		this.targets.clear();
	}
} as unknown as typeof ResizeObserver;

let target: HTMLDivElement;
let component: ReturnType<typeof mount> | null = null;

function mountTable(overrides: Record<string, unknown> = {}) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = mount(DataTable, {
		target,
		props: {
			columns,
			rows: ROWS,
			getRowId: (row: any) => row.id,
			...overrides
		}
	});
	flushSync();
}

afterEach(() => {
	if (component) {
		unmount(component);
		component = null;
	}
	target?.remove();
});

describe('DataTable', () => {
	it('renders one row per item using each column accessor', () => {
		mountTable();
		expect(target.textContent).toContain('admin');
		expect(target.textContent).toContain('admin@potionui.local');
		expect(target.textContent).toContain('someuser');
	});

	it('hides priority-2 columns unless detail is set', () => {
		mountTable();
		expect(target.textContent).not.toContain('secret');
		unmount(component!);
		component = null;
		target.remove();

		mountTable({ detail: true });
		expect(target.textContent).toContain('secret');
	});

	it('cycles sort asc -> desc -> off on repeated header clicks', () => {
		const onSortChange = vi.fn();
		mountTable({ onSortChange });
		const header = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Username'))!;
		header.click();
		expect(onSortChange).toHaveBeenLastCalledWith({ key: 'name', dir: 'asc' });

		unmount(component!);
		component = null;
		target.remove();
		mountTable({ onSortChange, sort: { key: 'name', dir: 'asc' } });
		const header2 = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Username'))!;
		header2.click();
		expect(onSortChange).toHaveBeenLastCalledWith({ key: 'name', dir: 'desc' });
	});

	it('fires onRowClick when a row is clicked, but not when the checkbox is clicked', () => {
		const onRowClick = vi.fn();
		const onSelectedChange = vi.fn();
		mountTable({ onRowClick, selected: new Set<string>(), onSelectedChange });
		const rows = target.querySelectorAll('[role="row"]');
		(rows[0] as HTMLElement).click();
		expect(onRowClick).toHaveBeenCalledWith(ROWS[0]);

		onRowClick.mockClear();
		const checkbox = rows[1].querySelector('button[role="checkbox"]') as HTMLButtonElement;
		checkbox.click();
		expect(onRowClick).not.toHaveBeenCalled();
		expect(onSelectedChange).toHaveBeenCalledWith(new Set(['b']));
	});

	it('selects and clears the whole page from the header checkbox', () => {
		const onSelectedChange = vi.fn();
		mountTable({ selected: new Set<string>(), onSelectedChange });
		const headCheckbox = target.querySelector('[role="checkbox"][aria-label="Select all rows on this page"]') as HTMLButtonElement;
		headCheckbox.click();
		expect(onSelectedChange).toHaveBeenCalledWith(new Set(['a', 'b']));

		unmount(component!);
		component = null;
		target.remove();
		mountTable({ selected: new Set(['a', 'b']), onSelectedChange });
		const headCheckbox2 = target.querySelector('[role="checkbox"][aria-label="Select all rows on this page"]') as HTMLButtonElement;
		expect(headCheckbox2.getAttribute('aria-checked')).toBe('true');
		headCheckbox2.click();
		expect(onSelectedChange).toHaveBeenLastCalledWith(new Set());
	});

	it('renders the loading skeleton instead of rows while loading', () => {
		mountTable({ loading: true, loadingRowCount: 2 });
		expect(target.textContent).not.toContain('admin');
		expect(target.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
	});

	it('renders the empty state when there are no rows, and the filtered-empty state when isFiltered', () => {
		const emptyState = createRawSnippet(() => ({ render: () => `<p>Nothing here yet</p>` }));
		const filteredEmptyState = createRawSnippet(() => ({ render: () => `<p>No matches</p>` }));
		mountTable({ rows: [], emptyState, filteredEmptyState, isFiltered: false });
		expect(target.textContent).toContain('Nothing here yet');

		unmount(component!);
		component = null;
		target.remove();
		mountTable({ rows: [], emptyState, filteredEmptyState, isFiltered: true });
		expect(target.textContent).toContain('No matches');
	});

	it('renders only the desktop rows when wide and invokes the card snippet for no row', () => {
		const card = vi.fn(() => ({ render: () => `<span class="card-probe">card</span>`, setup: () => {} }));
		mountTable({ card: createRawSnippet(card as any), forceNarrow: false });
		expect(target.querySelectorAll('.dt-scroll > .dt-row:not(.dt-row--head)')).toHaveLength(2);
		expect(target.querySelector('.dt-mobile')).toBeNull();
		expect(target.querySelectorAll('.card-probe')).toHaveLength(0);
	});

	it('renders only one card per row and no grid rows or header when narrow', () => {
		const card = createRawSnippet((row: () => Row) => ({ render: () => `<span class="card-probe">${row().name}</span>` }));
		mountTable({ card, forceNarrow: true });
		expect(target.querySelectorAll('.dt-mobile > div')).toHaveLength(2);
		expect(target.querySelectorAll('.card-probe')).toHaveLength(2);
		expect(target.querySelector('.dt-row')).toBeNull();
	});
});

describe('DataTable measured width', () => {
	const widthDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');

	beforeEach(() => {
		width = 0;
		Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => width });
	});

	afterEach(() => {
		if (widthDescriptor) Object.defineProperty(HTMLElement.prototype, 'clientWidth', widthDescriptor);
	});

	const card = createRawSnippet((row: () => Row) => ({ render: () => `<span class="card-probe">${row().name}</span>` }));

	function resizeTo(next: number) {
		width = next;
		observers.forEach((o) => o.callback([...o.targets].map((el) => ({ target: el }))));
		flushSync();
	}

	it('shows cards at exactly 640px and the grid at 641px', async () => {
		width = 640;
		mountTable({ card });
		await Promise.resolve();
		flushSync();
		expect(target.querySelectorAll('.dt-mobile > div')).toHaveLength(2);
		expect(target.querySelector('.dt-row')).toBeNull();

		unmount(component!);
		component = null;
		target.remove();
		width = 641;
		mountTable({ card });
		await Promise.resolve();
		flushSync();
		expect(target.querySelector('.dt-mobile')).toBeNull();
		expect(target.querySelectorAll('.dt-row')).toHaveLength(3);
	});

	it('renders the desktop layout flagged as unmeasured while the width is 0', () => {
		mountTable({ card });
		expect(target.querySelector('.dt-root')!.hasAttribute('data-unmeasured')).toBe(true);
		expect(target.querySelectorAll('.dt-row')).toHaveLength(3);
		expect(target.querySelector('.dt-mobile')).toBeNull();
	});

	it('clears the unmeasured flag once a width is known and switches layout when the width changes', async () => {
		width = 900;
		mountTable({ card });
		await Promise.resolve();
		flushSync();
		expect(target.querySelector('.dt-root')!.hasAttribute('data-unmeasured')).toBe(false);
		expect(target.querySelector('.dt-mobile')).toBeNull();

		resizeTo(500);
		expect(target.querySelectorAll('.dt-mobile > div')).toHaveLength(2);
		expect(target.querySelector('.dt-row')).toBeNull();

		resizeTo(900);
		expect(target.querySelector('.dt-mobile')).toBeNull();
		expect(target.querySelectorAll('.dt-row')).toHaveLength(3);
	});
});
