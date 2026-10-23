import { deepEqual } from '$lib/stores/tabDraftOps';
import type { VariablesMap } from './variableDefs';

export function applyVariableEdits(current: VariablesMap, from: VariablesMap, to: VariablesMap): VariablesMap {
	const next: VariablesMap = { ...current };
	for (const name of Object.keys(from)) {
		if (!(name in to)) delete next[name];
	}
	for (const [name, def] of Object.entries(to)) {
		if (!(name in from) || !deepEqual(from[name], def)) next[name] = def;
	}
	return next;
}
