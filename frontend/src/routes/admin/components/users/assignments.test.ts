import { describe, it, expect, vi, beforeEach } from 'vitest';

const { success, loggerError } = vi.hoisted(() => ({ success: vi.fn(), loggerError: vi.fn() }));

vi.mock('$lib/stores/toast', () => ({ toasts: { success } }));
vi.mock('$lib/utils/logger', () => ({ logger: { error: loggerError } }));

import { applyAssignments, applySingle, ensureOk } from './assignments';

beforeEach(() => {
	vi.clearAllMocks();
});

describe('ensureOk', () => {
	it('passes on success', () => {
		expect(() => ensureOk({ success: true }, 'nope')).not.toThrow();
	});

	it('throws the server message, then the error, then the fallback', () => {
		expect(() => ensureOk({ success: false, message: 'm', error: 'e' }, 'f')).toThrow('m');
		expect(() => ensureOk({ success: false, error: 'e' }, 'f')).toThrow('e');
		expect(() => ensureOk({ success: false }, 'f')).toThrow('f');
	});
});

describe('applySingle', () => {
	it('runs the first added id', async () => {
		const run = vi.fn().mockResolvedValue(undefined);
		await applySingle({ add: ['a', 'b'], remove: [] }, run);
		expect(run).toHaveBeenCalledExactlyOnceWith('a');
	});

	it('does nothing for an empty add list', async () => {
		const run = vi.fn();
		await applySingle({ add: [], remove: ['x'] }, run);
		expect(run).not.toHaveBeenCalled();
	});
});

describe('applyAssignments', () => {
	it('assigns, unassigns, reloads once and toasts the change count', async () => {
		const assign = vi.fn().mockResolvedValue(undefined);
		const unassign = vi.fn().mockResolvedValue(undefined);
		const reload = vi.fn().mockResolvedValue(undefined);

		const result = await applyAssignments({ add: ['a'], remove: ['b'] }, { assign, unassign }, reload, 'presets');

		expect(assign).toHaveBeenCalledWith('a');
		expect(unassign).toHaveBeenCalledWith('b');
		expect(reload).toHaveBeenCalledOnce();
		expect(result.ok.sort()).toEqual(['a', 'b']);
		expect(success).toHaveBeenCalledWith('Updated presets: 2 changes applied');
	});

	it('skips reload and toast when every change fails', async () => {
		const assign = vi.fn().mockRejectedValue(new Error('boom'));
		const reload = vi.fn();

		const result = await applyAssignments({ add: ['a'], remove: [] }, { assign, unassign: vi.fn() }, reload, 'LLMs');

		expect(result.ok).toEqual([]);
		expect(result.failed).toHaveLength(1);
		expect(reload).not.toHaveBeenCalled();
		expect(success).not.toHaveBeenCalled();
	});

	it('still toasts when the reload fails', async () => {
		const reload = vi.fn().mockRejectedValue(new Error('reload'));

		await applyAssignments({ add: ['a'], remove: [] }, { assign: vi.fn().mockResolvedValue(undefined), unassign: vi.fn() }, reload, 'models');

		expect(loggerError).toHaveBeenCalled();
		expect(success).toHaveBeenCalledWith('Updated models: 1 change applied');
	});
});
