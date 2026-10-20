import type { GenerationHistoryItem } from '$lib/types/history';

export interface DeleteCopy {
	title: string;
	message: string;
}

export function gridDeleteMessage(cellCount: number): string {
	const noun = cellCount === 1 ? 'image' : 'images';
	return `Delete this X/Y compare? ${cellCount} ${noun} will be deleted.\n\nThis permanently removes every image of the compare from your disk. This action cannot be undone.`;
}

export function cellLabel(generation: GenerationHistoryItem): string {
	return Object.entries(generation.axis_values ?? {})
		.map(([field, value]) => `${field}: ${value}`)
		.join(', ');
}

export function deleteCopy(generation: GenerationHistoryItem): DeleteCopy {
	if (generation.grid) {
		return { title: 'Delete X/Y Compare', message: gridDeleteMessage(generation.grid.cell_count) };
	}
	if (generation.grid_id) {
		const label = cellLabel(generation);
		return {
			title: 'Delete Compare Cell',
			message: `Delete only this cell${label ? ` (${label})` : ''} of the X/Y compare? The other cells are kept. This action cannot be undone.`
		};
	}
	const fileCount = generation.files.length;
	return {
		title: 'Delete Generation',
		message:
			fileCount > 0
				? `Are you sure you want to delete this generation and all its files? This action cannot be undone.\n\nThis will permanently delete ${fileCount} generated file(s) from your disk.`
				: 'Are you sure you want to delete this generation and all its files? This action cannot be undone.'
	};
}

export interface BulkDeletePlan {
	gridIds: string[];
	looseIds: string[];
	imageCount: number;
}

export function planBulkDelete(selectedIds: string[], generations: GenerationHistoryItem[]): BulkDeletePlan {
	const byId = new Map(generations.map((generation) => [generation.id, generation]));
	const gridIds: string[] = [];
	const looseIds: string[] = [];
	let imageCount = 0;
	for (const id of selectedIds) {
		const grid = byId.get(id)?.grid;
		if (grid) {
			if (!gridIds.includes(grid.id)) {
				gridIds.push(grid.id);
				imageCount += grid.cell_count;
			}
		} else {
			looseIds.push(id);
			imageCount += 1;
		}
	}
	return { gridIds, looseIds, imageCount };
}

export function bulkDeleteMessage(plan: BulkDeletePlan): string {
	if (plan.gridIds.length === 0) {
		const count = plan.looseIds.length;
		return `Are you sure you want to delete ${count} generation(s) and all their files? This action cannot be undone.\n\nThis will permanently delete all selected generations and their files from your disk.`;
	}
	const compares = `${plan.gridIds.length} X/Y compare${plan.gridIds.length === 1 ? '' : 's'}`;
	const loose = plan.looseIds.length > 0 ? ` and ${plan.looseIds.length} other generation(s)` : '';
	return `Delete ${compares}${loose}? ${plan.imageCount} images will be deleted.\n\nThis permanently removes every image of each compare from your disk. This action cannot be undone.`;
}
