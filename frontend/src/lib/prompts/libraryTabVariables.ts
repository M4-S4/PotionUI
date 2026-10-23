import { get, readable, type Readable } from 'svelte/store';
import { activeTab, tabsStore } from '$lib/stores/tabs';
import type { Segment } from '$lib/types/segments';
import { conditionReadersFor } from '$lib/utils/conditionTags';
import { countUsagesByName } from '$lib/utils/promptVariables';
import type { ConditionReaders, VariableDef, VariablesMap } from '$lib/utils/variableDefs';

export interface LibraryTabVariables {
	variables: VariablesMap;
	texts: readonly string[];
	setDef: (name: string, def: VariableDef) => void;
	setAll: (variables: VariablesMap) => void;
}

export interface LibraryVariableScope {
	usageCounts: Record<string, number>;
	readers: ConditionReaders;
}

const NO_VARIABLES: VariablesMap = {};

function setDef(name: string, def: VariableDef): void {
	const tab = get(activeTab);
	tabsStore.updateTab(tab.id, { variables: { ...(tab.variables ?? NO_VARIABLES), [name]: def } });
}

function setAll(variables: VariablesMap): void {
	tabsStore.updateTab(get(activeTab).id, { variables });
}

export const libraryTabVariables: Readable<LibraryTabVariables> = readable<LibraryTabVariables>(
	{ variables: NO_VARIABLES, texts: [], setDef, setAll },
	(set) => {
		let seen: [VariablesMap, Segment[] | undefined, Segment[] | undefined] | null = null;
		return activeTab.subscribe((tab) => {
			const variables = tab.variables ?? NO_VARIABLES;
			if (seen && seen[0] === variables && seen[1] === tab.promptSegments && seen[2] === tab.negativePromptSegments) return;
			seen = [variables, tab.promptSegments, tab.negativePromptSegments];
			const segments = [...(tab.promptSegments ?? []), ...(tab.negativePromptSegments ?? [])];
			set({ variables, texts: segments.map((segment) => segment.content), setDef, setAll });
		});
	}
);

export function libraryVariableScope(source: LibraryTabVariables, draft: readonly string[]): LibraryVariableScope {
	const texts = [...source.texts, ...draft];
	return { usageCounts: countUsagesByName(texts), readers: conditionReadersFor(texts) };
}
