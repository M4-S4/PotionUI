import { logger } from '$lib/utils/logger';
import { toasts } from '$lib/stores/toast';
import { applyDiff } from '$lib/components/picker/applyDiff';
import type { ApplyResult, PickerDiff } from '$lib/components/picker/types';
import { pluralize } from './usersLibrary';

export interface AssignmentHandlers {
	assign: (id: string) => Promise<void>;
	unassign: (id: string) => Promise<void>;
}

export function ensureOk(result: { success: boolean; message?: string; error?: string }, fallback: string) {
	if (!result.success) throw new Error(result.message || result.error || fallback);
}

export async function applySingle(diff: PickerDiff, run: (id: string) => Promise<void>) {
	const id = diff.add[0];
	if (id) await run(id);
}

export async function applyAssignments(
	diff: PickerDiff,
	handlers: AssignmentHandlers,
	reload: () => Promise<void>,
	noun: string
): Promise<ApplyResult> {
	const result = await applyDiff(diff, handlers);
	if (result.ok.length > 0) {
		try {
			await reload();
		} catch (error) {
			logger.error(`Failed to reload ${noun}:`, error);
		}
		toasts.success(`Updated ${noun}: ${pluralize(result.ok.length, 'change')} applied`);
	}
	return result;
}
