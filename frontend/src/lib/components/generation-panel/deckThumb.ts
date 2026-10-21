import type { GenerationState } from '$lib/types/tabs';
import { entryFileType, galleryItemAt, galleryItemUrl, galleryTotal } from '$lib/components/workbench/workbenchGallery';

export type DeckThumbKind = 'image' | 'video' | 'audio' | 'mesh' | 'text' | 'other';

export interface DeckThumbPreset {
	id: string;
	name: string;
	cover?: string | null;
	category?: string | null;
}

export type DeckThumbView =
	| { tile: 'media'; kind: 'image' | 'video'; src: string; count: number; duration: number | null }
	| { tile: 'icon'; kind: DeckThumbKind; count: number; duration: number | null; pending: boolean }
	| { tile: 'live'; src: string }
	| { tile: 'cover'; preset: DeckThumbPreset; cover: string }
	| { tile: 'family'; preset: DeckThumbPreset | null };

export const DECK_THUMB_ICONS: Record<DeckThumbKind, string> = {
	image: 'photo',
	video: 'film',
	audio: 'audio',
	mesh: 'cube',
	text: 'document',
	other: 'layers'
};

export function kindForCategory(category: string | null | undefined): DeckThumbKind {
	if (category === 'image' || category === 'video' || category === 'audio') return category;
	if (category === '3d') return 'mesh';
	return 'other';
}

function duration(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

function flagged(value: Record<string, unknown> | null | undefined): boolean {
	return !!value && (value.content_flagged === true || value.preview_suppressed === true || value.nsfw === true);
}

export function deriveDeckThumb(input: {
	generation: GenerationState;
	isGenerating: boolean;
	preset: DeckThumbPreset | null;
}): DeckThumbView {
	const { generation, isGenerating, preset } = input;
	if (isGenerating) {
		const current = generation.currentGeneration as Record<string, unknown> | null;
		const preview = current?.current_image;
		if (typeof preview === 'string' && preview && !flagged(current)) return { tile: 'live', src: preview };
		return { tile: 'icon', kind: kindForCategory(preset?.category), count: 0, duration: null, pending: true };
	}
	const batches = {
		images: generation.batchImages,
		videos: generation.batchVideos,
		audios: generation.batchAudios,
		meshes: generation.batchMeshes
	};
	const total = galleryTotal(batches);
	if (total > 0) {
		const entry = galleryItemAt(batches, total - 1);
		const kind = (entryFileType(entry) || 'other') as DeckThumbKind;
		const count = total > 1 ? total : 0;
		const item = (entry?.item ?? null) as Record<string, unknown> | null;
		const src = galleryItemUrl(item);
		const length = duration(item?.duration);
		if ((kind === 'image' || kind === 'video') && src && !flagged(item)) return { tile: 'media', kind, src, count, duration: length };
		return { tile: 'icon', kind, count, duration: kind === 'audio' || kind === 'video' ? length : null, pending: false };
	}
	const outputs = Object.keys(generation.pluginOutputs ?? {});
	if (outputs.length > 0) {
		const kind: DeckThumbKind = outputs.some((key) => /text|caption|prompt/i.test(key)) ? 'text' : 'other';
		return { tile: 'icon', kind, count: 0, duration: null, pending: false };
	}
	if (preset?.cover) return { tile: 'cover', preset, cover: preset.cover };
	return { tile: 'family', preset };
}

export function hasDeckResult(view: DeckThumbView): boolean {
	return view.tile === 'media' || (view.tile === 'icon' && !view.pending);
}

export function formatClock(seconds: number): string {
	const whole = Math.round(seconds);
	const minutes = Math.floor(whole / 60);
	return `${minutes}:${String(whole % 60).padStart(2, '0')}`;
}
