import { describe, expect, it, vi } from 'vitest';
import { canReturnToDetails, registerDetailsReturn, returnToDetails } from './detailsReturn';

describe('detailsReturn', () => {
	it('does nothing and says so when no History page is listening', () => {
		expect(canReturnToDetails()).toBe(false);
		expect(returnToDetails({ generationId: 'g', fileIndex: 0 })).toBe(false);
	});

	it('hands the generation and file index to the listening page', () => {
		const handler = vi.fn();
		const unregister = registerDetailsReturn(handler);

		expect(canReturnToDetails()).toBe(true);
		expect(returnToDetails({ generationId: 'g1', fileIndex: 3 })).toBe(true);
		expect(handler).toHaveBeenCalledWith({ generationId: 'g1', fileIndex: 3 });
		unregister();
		expect(canReturnToDetails()).toBe(false);
	});

	it('keeps a newer listener when an older page unregisters late', () => {
		const first = vi.fn();
		const second = vi.fn();
		const unregisterFirst = registerDetailsReturn(first);
		registerDetailsReturn(second);

		unregisterFirst();
		returnToDetails({ generationId: 'g', fileIndex: 0 });

		expect(second).toHaveBeenCalledTimes(1);
		expect(first).not.toHaveBeenCalled();
	});
});
