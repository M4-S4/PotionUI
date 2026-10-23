import type { IndexingDigests, IndexingLevelProfile, IndexingPendingItem, IndexingStatus } from '$lib/services/api/models';

export const MIB = 1024 * 1024;
export const GIB = MIB * 1024;

export const LEVELS: IndexingLevelProfile[] = [
	{ id: 'low', workers: 1, rate_bps: 96 * MIB, pauses_for_generation: true },
	{ id: 'normal', workers: 2, rate_bps: 384 * MIB, pauses_for_generation: true },
	{ id: 'max', workers: 4, rate_bps: null, pauses_for_generation: false }
];

export function digests(overrides: Partial<IndexingDigests> = {}): IndexingDigests {
	return {
		policy: 'background',
		level: 'low',
		levels: LEVELS,
		paused: false,
		pause_reason: null,
		busy: false,
		workers: 1,
		pending_total: 0,
		queued: 0,
		deferred: 0,
		ready: 0,
		ready_bytes: 0,
		pending_bytes: 0,
		queued_bytes: 0,
		current: [],
		throughput_bps: null,
		eta_seconds: null,
		duplicate_checks: 0,
		failed: [],
		plan: { awaiting_choice: false, files: 0, bytes: 0, min_files: 20, min_bytes: 64 * GIB },
		version: 1,
		...overrides
	};
}

export function status(digestOverrides: Partial<IndexingDigests> = {}, overrides: Partial<IndexingStatus> = {}): IndexingStatus {
	return { state: 'done', found_on_disk: 487, indexed: 0, digests: digests(digestOverrides), ...overrides };
}

export const RUNNING = {
	policy: 'background' as const,
	ready: 218,
	pending_total: 269,
	queued: 269,
	ready_bytes: 32 * GIB,
	pending_bytes: 1700 * GIB,
	queued_bytes: 1700 * GIB,
	throughput_bps: 96 * MIB,
	eta_seconds: 5 * 3600 + 12 * 60,
	duplicate_checks: 3,
	current: [{ model_id: 'm1', filename: 'neon_wan_lora_v6-190.safetensors', size: 308 * MIB, done: 279 * MIB }]
};

export const AWAITING = {
	policy: 'unset' as const,
	ready: 15,
	pending_total: 472,
	deferred: 472,
	pending_bytes: 1.75 * 1024 * GIB,
	plan: { awaiting_choice: true, files: 472, bytes: 1.75 * 1024 * GIB, min_files: 20, min_bytes: 64 * GIB }
};

export function pendingItem(id: string, filename: string, model_type: string, size: number): IndexingPendingItem {
	return {
		model_id: id,
		location_id: `loc-${id}`,
		filename,
		model_type,
		size,
		root_label: 'ComfyUI models',
		rel_path: filename,
		queued: false
	};
}
