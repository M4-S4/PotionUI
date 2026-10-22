import { describe, it, expect } from 'vitest';
import { PANE_HANDLE_STRIP, PANE_HANDLE_GRIP } from './paneHandle';
import { FORM_RAIL_WIDTH, PANE_HANDLE_WIDTH } from '$lib/stores/generationLayout';

const tokens = (classes: string) => classes.split(/\s+/);

describe('pane handle strip', () => {
	it('is the 12px surface-3 strip with a 1px line that the form handle uses', () => {
		expect(tokens(PANE_HANDLE_STRIP)).toEqual(
			expect.arrayContaining(['h-full', 'w-3', 'flex-shrink-0', 'border-r', 'border-line', 'bg-surface-3', 'hover:bg-line-hover'])
		);
	});

	it('is as wide in the layout maths as the form rail, and w-3 is 12px', () => {
		expect(PANE_HANDLE_WIDTH).toBe(FORM_RAIL_WIDTH);
		expect(PANE_HANDLE_WIDTH).toBe(12);
	});

	it('shows the dragging state with the same fill as the hover state', () => {
		expect(tokens(PANE_HANDLE_STRIP)).toContain('data-[dragging=true]:bg-line-hover');
	});
});

describe('pane handle grip', () => {
	it('is a 4 by 40px pill centred on the strip that takes no pointer events', () => {
		expect(tokens(PANE_HANDLE_GRIP)).toEqual(
			expect.arrayContaining(['absolute', 'left-1/2', 'top-1/2', '-translate-x-1/2', '-translate-y-1/2', 'h-10', 'w-1', 'pointer-events-none'])
		);
	});

	it('brightens on hover and while dragging, with tokens only', () => {
		expect(tokens(PANE_HANDLE_GRIP)).toEqual(expect.arrayContaining(['group-hover:bg-fg-muted', 'group-data-[dragging=true]:bg-fg']));
		expect(PANE_HANDLE_GRIP + PANE_HANDLE_STRIP).not.toMatch(/#[0-9a-f]{3,8}\b|zinc-|gray-|dark:/i);
	});
});
