import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { WORKBENCH_EMPTY_LAYOUTS, WORKBENCH_PANE_PADDING, workbenchEmptyState, type WorkbenchEmptyLayout } from './workbenchEmpty';

const idle = { hasDisplayMedia: false, status: null, previewSuppressed: false, contentBlocked: null };

describe('workbenchEmptyState', () => {
	it('is empty with nothing to show, mid-run before the first preview, and after a cancel', () => {
		expect(workbenchEmptyState(idle)).toBe('empty');
		expect(workbenchEmptyState({ ...idle, status: 'running' })).toBe('empty');
		expect(workbenchEmptyState({ ...idle, status: 'cancelled' })).toBe('empty');
	});

	it('is null once there is media to display', () => {
		expect(workbenchEmptyState({ ...idle, hasDisplayMedia: true })).toBeNull();
		expect(workbenchEmptyState({ ...idle, hasDisplayMedia: true, status: 'completed' })).toBeNull();
		expect(workbenchEmptyState({ ...idle, hasDisplayMedia: true, status: 'failed' })).toBeNull();
	});

	it('names a failure, a suppressed preview and blocked content', () => {
		expect(workbenchEmptyState({ ...idle, status: 'failed' })).toBe('failed');
		expect(workbenchEmptyState({ ...idle, previewSuppressed: true })).toBe('suppressed');
		expect(workbenchEmptyState({ ...idle, contentBlocked: { blocked_count: 1, total: 2 } })).toBe('blocked');
	});

	it('puts a failure first and a suppressed preview before blocked content', () => {
		const everything = { ...idle, previewSuppressed: true, contentBlocked: { blocked_count: 1, total: 2 } };
		expect(workbenchEmptyState({ ...everything, status: 'failed' })).toBe('failed');
		expect(workbenchEmptyState(everything)).toBe('suppressed');
	});
});

describe('workbench empty layouts', () => {
	const layouts = Object.keys(WORKBENCH_EMPTY_LAYOUTS) as WorkbenchEmptyLayout[];

	it('reuses the app dot grid in every layout', () => {
		for (const layout of layouts) expect(WORKBENCH_EMPTY_LAYOUTS[layout].dots.split(' ')).toContain('dot-grid');
	});

	it('fades the dots only in the layout that sits above the prompt segments', () => {
		const fading = layouts.filter((layout) => WORKBENCH_EMPTY_LAYOUTS[layout].dots.split(' ').includes('dot-grid-fade'));
		expect(fading).toEqual(['fade']);
	});

	it('fills the pane in the pane layout and keeps its own height in the others', () => {
		expect(WORKBENCH_EMPTY_LAYOUTS.pane.sized).toBe(false);
		expect(WORKBENCH_EMPTY_LAYOUTS.pane.slider).toBe(false);
		expect(WORKBENCH_EMPTY_LAYOUTS.fade.sized && WORKBENCH_EMPTY_LAYOUTS.window.sized).toBe(true);
	});

	it('keeps the pane root static and reserves the stage height only when something follows it', () => {
		const pane = WORKBENCH_EMPTY_LAYOUTS.pane;
		expect(pane.root.split(' ')).not.toContain('relative');
		expect(pane.root.split(' ')).toEqual(expect.arrayContaining(['group/workbench', '[&:has(~*)]:min-h-[var(--workbench-stage)]']));
		expect(pane.reserve).toBe(true);
		expect(pane.stage.split(' ')).toContain('top-[var(--workbench-stage-top,0px)]');
		expect(pane.copy.split(' ')).toContain('pointer-events-auto');
		expect(pane.copy.split(' ')).toContain('group-[:has(~*)]/workbench:h-[min(100%,var(--workbench-region))]');
		expect(WORKBENCH_EMPTY_LAYOUTS.fade.reserve || WORKBENCH_EMPTY_LAYOUTS.window.reserve).toBe(false);
		expect(WORKBENCH_EMPTY_LAYOUTS.fade.root).toBe('relative');
		expect(WORKBENCH_EMPTY_LAYOUTS.window.root).toBe('relative');
	});

	it('scrolls the copy and centres it only while it fits, in every layout', () => {
		for (const layout of layouts) {
			const copy = WORKBENCH_EMPTY_LAYOUTS[layout].copy.split(' ');
			expect(copy).toEqual(expect.arrayContaining(['overflow-y-auto', '[justify-content:safe_center]', 'items-center', 'text-center']));
			expect(copy).not.toContain('justify-center');
		}
	});

	it('names the pane padding the stage offset adds below a notice', () => {
		expect(WORKBENCH_PANE_PADDING).toBe(16);
	});

	it('uses no raw colours and no text below 12px', () => {
		for (const layout of layouts) {
			const all = Object.values(WORKBENCH_EMPTY_LAYOUTS[layout]).join(' ');
			expect(all).not.toMatch(/#[0-9a-f]{3,8}\b|zinc-|gray-|text-xs|text-2xs/i);
		}
	});
});

describe('dot grid fade', () => {
	const css = readFileSync(new URL('../../../app.css', import.meta.url), 'utf8');
	const rule = (selector: string) => css.match(new RegExp(`\\n\\s*\\${selector} \\{([^}]*)\\}`))?.[1] ?? '';

	it('masks the dots from opaque to transparent, top to bottom, with token colours only', () => {
		const fade = rule('.dot-grid-fade');
		expect(fade).toMatch(/mask-image: linear-gradient\(to bottom, rgb\(var\(--fg\)\) \d+%, transparent 100%\)/);
		expect(fade).not.toMatch(/#[0-9a-f]{3,8}\b/i);
	});

	it('keeps the original dot grid untouched', () => {
		const dots = rule('.dot-grid');
		expect(dots).toContain('radial-gradient(rgb(var(--fg) / 0.06) 1px, transparent 1px)');
		expect(dots).toContain('background-size: 16px 16px');
	});
});
