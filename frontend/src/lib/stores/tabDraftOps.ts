import type { TabDraftOp } from '$lib/services/api/tabDrafts';
import {
	normalizeVariableDef,
	type ChoiceVariableMode,
	type VariableDef,
	type VariableOption,
	type VariablesMap
} from '$lib/utils/variableDefs';

export function deepEqual(a: unknown, b: unknown): boolean {
	if (Object.is(a, b)) return true;
	if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
	if (Array.isArray(a) !== Array.isArray(b)) return false;
	if (Array.isArray(a) && Array.isArray(b)) {
		return a.length === b.length && a.every((item, index) => deepEqual(item, b[index]));
	}
	const left = a as Record<string, unknown>;
	const right = b as Record<string, unknown>;
	const keys = Object.keys(left).filter((key) => left[key] !== undefined);
	const rightKeys = Object.keys(right).filter((key) => right[key] !== undefined);
	if (keys.length !== rightKeys.length) return false;
	return keys.every((key) => deepEqual(left[key], right[key]));
}

export function flattenDef(stored: VariablesMap[string]): Record<string, unknown> {
	const def = normalizeVariableDef(stored);
	if (def.type === 'choice') {
		return {
			type: 'choice',
			options: def.options,
			mode: def.mode,
			pinnedIndex: def.mode === 'pin' ? (def.pinnedIndex ?? null) : null
		};
	}
	if (def.type === 'number') {
		return {
			type: 'number',
			mode: def.mode,
			min: def.min,
			max: def.max,
			step: def.step,
			value: def.value
		};
	}
	return { type: 'text', value: def.value };
}

export function sameForSync(a: VariablesMap[string], b: VariablesMap[string]): boolean {
	return deepEqual(flattenDef(a), flattenDef(b));
}

export function setOp(name: string, stored: VariablesMap[string]): TabDraftOp {
	const def = normalizeVariableDef(stored);
	if (def.type === 'choice') {
		return {
			op: 'set',
			name,
			type: 'choice',
			options: def.options,
			mode: def.mode,
			pinned_index: def.mode === 'pin' ? (def.pinnedIndex ?? null) : null
		};
	}
	if (def.type === 'number') {
		return {
			op: 'set',
			name,
			type: 'number',
			mode: def.mode,
			min: def.min,
			max: def.max,
			step: def.step,
			value: def.value
		};
	}
	return { op: 'set', name, type: 'text', value: def.value };
}

export function removeOp(name: string): TabDraftOp {
	return { op: 'remove', name };
}

export function defFromOp(op: TabDraftOp): VariableDef {
	if (op.type === 'choice') {
		const mode = (op.mode as ChoiceVariableMode | undefined) ?? 'shuffle';
		return {
			type: 'choice',
			options: (op.options as VariableOption[] | undefined) ?? ['', ''],
			mode,
			pinnedIndex: mode === 'pin' ? (op.pinned_index ?? null) : null
		};
	}
	if (op.type === 'number') {
		return {
			type: 'number',
			mode: (op.mode as ChoiceVariableMode | undefined) ?? 'shuffle',
			min: op.min ?? 0,
			max: op.max ?? 0,
			step: op.step ?? 1,
			value: typeof op.value === 'number' ? op.value : (op.min ?? 0)
		};
	}
	return { type: 'text', value: typeof op.value === 'string' ? op.value : '' };
}

export function diffVariables(prev: VariablesMap, next: VariablesMap): TabDraftOp[] {
	const ops: TabDraftOp[] = [];
	for (const name of Object.keys(prev)) {
		if (!(name in next)) ops.push(removeOp(name));
	}
	for (const [name, def] of Object.entries(next)) {
		if (!(name in prev) || !sameForSync(prev[name], def)) ops.push(setOp(name, def));
	}
	return ops;
}

export function overlayOps(base: VariablesMap, ops: Iterable<TabDraftOp>): VariablesMap {
	const next: VariablesMap = { ...base };
	for (const op of ops) {
		if (op.op === 'remove') delete next[op.name];
		else next[op.name] = defFromOp(op);
	}
	return next;
}
