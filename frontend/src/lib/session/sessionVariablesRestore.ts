import type { VariablesMap } from '$lib/utils/variableDefs';

export type SessionVariablesDecision = 'replace' | 'keep' | 'defer';

export function decideSessionVariables(
	ready: boolean,
	currentVariables: VariablesMap | undefined
): SessionVariablesDecision {
	if (!ready) return 'defer';
	return Object.keys(currentVariables ?? {}).length > 0 ? 'keep' : 'replace';
}

export interface DeferredSessionVariablesDeps {
	settled: () => Promise<boolean>;
	currentVariables: () => VariablesMap | undefined;
	apply: (variables: VariablesMap) => void;
	replace: (variables: VariablesMap) => void;
}

export async function finishDeferredSessionVariables(
	deps: DeferredSessionVariablesDeps,
	sessionVariables: VariablesMap
): Promise<SessionVariablesDecision> {
	const ready = await deps.settled();
	const decision = decideSessionVariables(ready, deps.currentVariables());
	if (decision === 'replace') {
		deps.apply(sessionVariables);
		deps.replace(sessionVariables);
	}
	return decision;
}
