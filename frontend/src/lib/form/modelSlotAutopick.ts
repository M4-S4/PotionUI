import type { FieldConfig } from '$lib/form/reactions';
import { fromComponentValue } from '$lib/utils/presetFormOverrides';
import { refFor } from '$lib/utils/modelRef';
import { buildModelSearchRequest } from '$lib/utils/modelSearchParams';

export interface AutopickSlot {
	name: string;
	presetId: string;
	modelType: string;
	modelTypes: string[];
	unquantizedOnly: boolean;
	filterTagIds: string[];
	tagFilters: string[];
}

export interface AutopickSearchResult {
	models: any[];
	total?: number;
}

export interface AutopickDecisionInput {
	hasValue: boolean;
	restored: boolean;
	cleared: boolean;
	result: AutopickSearchResult | null | undefined;
}

export function hasModelValue(field: Pick<FieldConfig, 'type'>, value: unknown): boolean {
	const wire = fromComponentValue(field.type, value);
	return typeof wire === 'string' && wire.trim() !== '';
}

export function decideAutopick(input: AutopickDecisionInput): any | null {
	if (input.hasValue || input.restored || input.cleared) return null;
	const models = input.result?.models;
	if (!Array.isArray(models) || models.length !== 1) return null;
	if (typeof input.result?.total === 'number' && input.result.total !== 1) return null;
	return models[0]?.id ? models[0] : null;
}

export function collectAutopickSlots(
	fields: FieldConfig[],
	data: Record<string, unknown>,
	presetId: string
): AutopickSlot[] {
	const slots: AutopickSlot[] = [];
	const visit = (list: FieldConfig[]) => {
		for (const field of list ?? []) {
			if (
				field.name &&
				field.type === 'model' &&
				field.required &&
				field.visible !== false &&
				!field.readonly &&
				!field.disabled &&
				!hasModelValue(field, data[field.name])
			) {
				const configuration = field.configuration ?? {};
				const modelTypes = Array.isArray(configuration.model_types) ? configuration.model_types : [];
				const scopedPresetId = (field as { preset_id?: string }).preset_id || presetId;
				if (scopedPresetId) {
					const current = data[field.name] as { tagFilters?: string[] } | undefined;
					slots.push({
						name: field.name,
						presetId: scopedPresetId,
						modelType: configuration.model_type || modelTypes[0] || 'checkpoint',
						modelTypes,
						unquantizedOnly: configuration.unquantized_only === true,
						filterTagIds: Array.isArray(configuration.filter_tags) ? configuration.filter_tags : [],
						tagFilters: current?.tagFilters ?? configuration.tags ?? []
					});
				}
			}
			if (Array.isArray(field.children)) visit(field.children);
		}
	};
	visit(fields);
	return slots;
}

export function resolveSlotTagIds(
	names: string[],
	tags: Array<{ id?: string; name?: string }> | null | undefined
): string[] | null {
	if (names.length === 0) return [];
	if (!tags) return null;
	const ids: string[] = [];
	for (const name of names) {
		const id = tags.find((t) => t.name === name)?.id;
		if (!id) return null;
		ids.push(id);
	}
	return ids;
}

export function isSlotClearEdit(isModelField: boolean, previous: unknown, next: unknown): boolean {
	if (!isModelField) return false;
	const field = { type: 'model' };
	return hasModelValue(field, previous) && !hasModelValue(field, next);
}

export function autopickValue(slot: AutopickSlot, model: { id?: string | null }) {
	return { modelPath: refFor(model), tagFilters: slot.tagFilters };
}

export function slotSearchRequest(slot: AutopickSlot, tagIds: string[]) {
	return buildModelSearchRequest({
		modelType: slot.modelType,
		presetId: slot.presetId,
		limit: 2,
		tagIds,
		anyTagIds: slot.filterTagIds,
		modelTypes: slot.modelTypes,
		unquantizedOnly: slot.unquantizedOnly
	});
}

const clearedByTab = new Map<string, Set<string>>();

export function markSlotCleared(tabKey: string, name: string): void {
	let set = clearedByTab.get(tabKey);
	if (!set) {
		set = new Set();
		clearedByTab.set(tabKey, set);
	}
	set.add(name);
}

export function isSlotCleared(tabKey: string, name: string): boolean {
	return clearedByTab.get(tabKey)?.has(name) ?? false;
}

export function forgetClearedSlots(tabKey?: string): void {
	if (tabKey === undefined) clearedByTab.clear();
	else clearedByTab.delete(tabKey);
}

export interface AutopickSyncInput {
	loadKey: string;
	slots: AutopickSlot[];
	isPinned: (name: string) => boolean;
	isEmpty: (name: string) => boolean;
	onPick: (slot: AutopickSlot, model: any) => void;
}

export interface ModelSlotAutopick {
	sync(input: AutopickSyncInput): void;
	dispose(): void;
}

export interface ModelSlotAutopickDeps {
	tabKey: string;
	search: (slot: AutopickSlot) => Promise<AutopickSearchResult | null>;
	debounceMs?: number;
}

export function createModelSlotAutopick(deps: ModelSlotAutopickDeps): ModelSlotAutopick {
	const debounceMs = deps.debounceMs ?? 150;
	let currentLoadKey = '';
	let requested = new Set<string>();
	let queue: Array<{ slot: AutopickSlot; input: AutopickSyncInput }> = [];
	let timer: ReturnType<typeof setTimeout> | undefined;
	let disposed = false;

	async function run(slot: AutopickSlot, input: AutopickSyncInput) {
		let result: AutopickSearchResult | null = null;
		try {
			result = await deps.search(slot);
		} catch {
			result = null;
		}
		if (disposed || input.loadKey !== currentLoadKey) return;
		const model = decideAutopick({
			hasValue: !input.isEmpty(slot.name),
			restored: input.isPinned(slot.name),
			cleared: isSlotCleared(deps.tabKey, slot.name),
			result
		});
		if (model) input.onPick(slot, model);
	}

	function flush() {
		timer = undefined;
		const batch = queue;
		queue = [];
		for (const { slot, input } of batch) void run(slot, input);
	}

	return {
		sync(input) {
			if (disposed) return;
			if (input.loadKey !== currentLoadKey) {
				currentLoadKey = input.loadKey;
				requested = new Set();
				queue = [];
			}
			for (const slot of input.slots) {
				if (input.isPinned(slot.name) || isSlotCleared(deps.tabKey, slot.name)) continue;
				if (requested.has(slot.name)) continue;
				requested.add(slot.name);
				queue.push({ slot, input });
			}
			if (queue.length > 0 && timer === undefined) timer = setTimeout(flush, debounceMs);
		},
		dispose() {
			disposed = true;
			queue = [];
			if (timer !== undefined) clearTimeout(timer);
			timer = undefined;
		}
	};
}
