import { writable } from 'svelte/store';
import { loadMediaToolCatalog, mediaToolCatalog } from '$lib/media-tools/catalog';
import { syncCatalogTools } from './catalogTools';

export const mediaToolRegistrations = writable(0);

let mirroring = false;

function mirrorCatalog(): void {
	if (mirroring) return;
	mirroring = true;
	mediaToolCatalog.subscribe((tools) => {
		syncCatalogTools(tools);
		mediaToolRegistrations.update((tick) => tick + 1);
	});
}

export async function loadPluginMediaTools(): Promise<void> {
	mirrorCatalog();
	await loadMediaToolCatalog();
}
