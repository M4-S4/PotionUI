import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), 'utf8');
const panels = read('../../../routes/generate/components/GenerationPanels.svelte');
const workbench = read('../Workbench.svelte');
const pane = read('../../../routes/generate/components/GenerationWorkbenchPane.svelte');
const style = panels.slice(panels.indexOf('<style>'));

describe('emptyLayout prop', () => {
	it('is required on the workbench and on its pane wrapper, with no fallback layout', () => {
		expect(workbench).toMatch(/export let emptyLayout: WorkbenchEmptyLayout;/);
		expect(pane).toMatch(/export let emptyLayout: WorkbenchEmptyLayout;/);
		expect(workbench + pane).not.toMatch(/emptyLayout[^;\n]*=\s*'/);
	});
});

describe('GenerationPanels resize handle source', () => {
	it('keeps only touch-action in the scoped handle style, since the shared classes carry hover and drag', () => {
		expect(style).toMatch(/\.resize-handle\s*\{\s*touch-action: none;\s*\}/);
		expect(style).not.toMatch(/\.resize-handle:(hover|active)/);
		expect(style).not.toContain('background-color');
	});

	it('takes the workbench floor and the handle width from the layout module instead of local numbers', () => {
		expect(panels).toMatch(/WORKBENCH_FLOOR_WIDTH \+ PANE_HANDLE_WIDTH\}px/);
		expect(panels).not.toMatch(/WORKBENCH_MIN_WIDTH|RESIZE_HANDLE_WIDTH|'328px'/);
		expect(panels).not.toMatch(/WORKBENCH_FLOOR_WIDTH \+ PANE_HANDLE_WIDTH \+ \d/);
	});
});
