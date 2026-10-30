import type { ComponentType } from 'svelte';
import type { RecentLook } from '$lib/filters/recentLooks';
import { mediaToolRegistrations } from '$lib/tools/pluginTools';
import { registerMediaTool, unregisterMediaTools } from '$lib/tools/tools';
import { applyLookAvailability } from './editImage';
import HistoryApplyLookModal from '../components/HistoryApplyLookModal.svelte';

export function recentLookToolId(lookId: string): string {
	return `look:${lookId}`;
}

export function syncRecentLookTools(looks: readonly RecentLook[]): void {
	unregisterMediaTools((tool) => !!tool.lookId);
	for (const look of looks) {
		registerMediaTool({
			id: recentLookToolId(look.id),
			label: look.name,
			description: `Apply ${look.name} to this image in the image editor`,
			icon: 'palette',
			category: 'effects',
			source: 'core',
			scopes: ['history', 'library'],
			selection: { min: 1 },
			kinds: ['image'],
			lookId: look.id,
			componentProps: { lookId: look.id },
			note: look.group || undefined,
			applies: applyLookAvailability,
			component: HistoryApplyLookModal as unknown as ComponentType
		});
	}
	mediaToolRegistrations.update((tick) => tick + 1);
}
