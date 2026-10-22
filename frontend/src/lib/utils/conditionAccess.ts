import type { VariableDef, VariablesMap } from '$lib/utils/variableDefs';

export function conditionsAvailable(
	variables: VariablesMap,
	onVariableDefChange: ((name: string, def: VariableDef) => void) | undefined
): boolean {
	return onVariableDefChange !== undefined || Object.keys(variables).length > 0;
}
