import { applyLookAvailability, editImageAvailability } from './editImage';
import {
	listToolGroups,
	type MediaTool,
	type MediaToolContext,
	type MediaToolItem,
	type MediaToolScope
} from './tools';

export type RowToolAction = 'catalog' | 'edit-image' | 'apply-look';

export interface RowTool {
	id: string;
	label: string;
	icon: string;
	action: RowToolAction;
	enabled: boolean;
	reason: string;
	note: string;
	tool?: MediaTool;
}

export interface RowModel {
	editButtons: RowTool[];
	moreTools: RowTool[];
	effects: RowTool[];
}

export function buildRowContext(
	scope: Extract<MediaToolScope, 'history' | 'library'>,
	items: MediaToolItem[]
): MediaToolContext {
	return {
		scope,
		items,
		kinds: new Set(items.map((item) => item.kind)),
		collectionId: null,
		generations: [],
		generationIds: [],
		files: [],
		entry: { status: 'completed', stackCells: null, looseCell: false, error: null }
	};
}

export function buildRowModel(ctx: MediaToolContext): RowModel {
	const catalog: RowTool[] = [];
	for (const group of listToolGroups(ctx)) {
		if (group.category.id !== 'edit') continue;
		for (const { tool, availability } of group.tools) {
			if (availability.hidden) continue;
			catalog.push({
				id: tool.id,
				label: tool.label,
				icon: tool.icon,
				action: 'catalog',
				enabled: availability.enabled,
				reason: availability.reason ?? '',
				note: tool.note ?? '',
				tool
			});
		}
	}

	const editButtons = catalog.filter((entry) => entry.tool?.builtInEditor);
	const editImage = editImageAvailability(ctx);
	if (ctx.items.length > 0 && ctx.items.every((item) => item.kind === 'image')) {
		editButtons.push({
			id: 'edit-image',
			label: 'Edit image',
			icon: 'paint-brush',
			action: 'edit-image',
			enabled: editImage.enabled,
			reason: editImage.reason ?? '',
			note: ''
		});
	}

	const look = applyLookAvailability(ctx);
	const effects: RowTool[] = look.hidden
		? []
		: [
				{
					id: 'apply-look',
					label: 'Apply a look…',
					icon: 'palette',
					action: 'apply-look',
					enabled: look.enabled,
					reason: look.reason ?? '',
					note: ''
				}
			];

	return {
		editButtons,
		moreTools: catalog.filter((entry) => !entry.tool?.builtInEditor),
		effects
	};
}
