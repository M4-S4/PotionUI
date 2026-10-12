import type { ImageData, VideoData } from '$lib/types/tabs';

type PromptTagged = Pick<ImageData | VideoData, 'promptIndex' | 'promptText' | 'promptImage'>;

export interface PromptEntry<T extends PromptTagged> {
	item: T;
	index: number;
	kind: 'image' | 'video';
}

export interface PromptGroupTile<T extends PromptTagged> extends PromptEntry<T> {
	imageNumber: number;
}

export interface PromptGroup<T extends PromptTagged> {
	promptIndex: number;
	promptText: string;
	tiles: PromptGroupTile<T>[];
}

export function groupByPrompt<T extends PromptTagged>(entries: PromptEntry<T>[]): PromptGroup<T>[] {
	const groups = new Map<number, PromptGroup<T>>();
	for (const entry of entries) {
		const promptIndex = entry.item.promptIndex;
		if (typeof promptIndex !== 'number') continue;
		let group = groups.get(promptIndex);
		if (!group) {
			group = { promptIndex, promptText: entry.item.promptText ?? '', tiles: [] };
			groups.set(promptIndex, group);
		}
		if (!group.promptText && entry.item.promptText) group.promptText = entry.item.promptText;
		group.tiles.push({ ...entry, imageNumber: group.tiles.length + 1 });
	}
	return [...groups.values()].sort((a, b) => a.promptIndex - b.promptIndex);
}

export function isPromptBatch(items: PromptTagged[]): boolean {
	const seen = new Set<number>();
	for (const item of items) {
		if (typeof item.promptIndex === 'number') seen.add(item.promptIndex);
	}
	return seen.size > 1;
}

export interface BatchCaption {
	title: string;
	prompt: string;
}

export function batchCaption(items: PromptTagged[], item: PromptTagged | null | undefined): BatchCaption | null {
	if (!item || typeof item.promptIndex !== 'number' || !isPromptBatch(items)) return null;
	const siblings = items.filter((other) => other.promptIndex === item.promptIndex);
	const position = Math.max(1, siblings.indexOf(item as PromptTagged) + 1);
	return {
		title: `Prompt ${item.promptIndex + 1} · image ${position} of ${siblings.length}`,
		prompt: item.promptText ?? ''
	};
}
