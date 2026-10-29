import { get } from 'svelte/store';
import type { PaintSession } from '$lib/components/imageEditor/session';
import { paintFilters } from '$lib/components/imageEditor/registries';
import { filterCatalog, loadFilterCatalog } from './catalog';
import { sharedExtensions } from './extensions';
import { selectFilterItem } from './actions';
import { isLocked } from './strip';

export async function preselectLook(session: PaintSession, lookId: string): Promise<boolean> {
	await loadFilterCatalog();
	const catalog = get(filterCatalog);
	const item = catalog.items.find((candidate) => candidate.id === lookId);
	if (!item) return false;
	if (isLocked(item, sharedExtensions(catalog.ops, paintFilters.list()))) return false;
	try {
		await selectFilterItem(session, item);
	} catch {
		return false;
	}
	return true;
}
