import type { Tab } from '$lib/types/tabs';

export function contextToolsAvailable(tab: Pick<Tab, 'selectedPreset'> | null | undefined): boolean {
	return !!tab?.selectedPreset;
}
