import { describe, expect, it } from 'vitest';
import { FILTERS_TOOL, lookTool, parseInitialTool } from './initialTool';

describe('parseInitialTool', () => {
	it('reads a plain tool id', () => {
		expect(parseInitialTool('filters')).toEqual({ tool: 'filters', look: null });
	});

	it('reads a tool with a look after it', () => {
		expect(parseInitialTool('filters:ember')).toEqual({ tool: 'filters', look: 'ember' });
	});

	it('keeps a look id that has a colon in it, as a plugin look does', () => {
		expect(parseInitialTool('filters:retro-tape:vhs')).toEqual({ tool: 'filters', look: 'retro-tape:vhs' });
	});

	it('treats an empty look as none', () => {
		expect(parseInitialTool('filters:')).toEqual({ tool: 'filters', look: null });
	});

	it('round-trips what lookTool builds', () => {
		expect(parseInitialTool(lookTool('retro-tape:vhs'))).toEqual({ tool: FILTERS_TOOL, look: 'retro-tape:vhs' });
	});
});
