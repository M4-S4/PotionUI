import { describe, expect, it } from 'vitest';
import { KEEP_OPEN_MIN_WIDTH, effectiveKeepOpen } from './drawerBehavior';

describe('drawer behavior', () => {
	it('only keeps the drawer open on wide viewports', () => {
		expect(effectiveKeepOpen(true, KEEP_OPEN_MIN_WIDTH)).toBe(true);
		expect(effectiveKeepOpen(true, KEEP_OPEN_MIN_WIDTH - 1)).toBe(false);
		expect(effectiveKeepOpen(false, 2000)).toBe(false);
	});
});
