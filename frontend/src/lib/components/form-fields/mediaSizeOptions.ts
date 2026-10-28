import { fetchMediaMetadata, type MetadataSource } from './mediaLoaderMetadata';
import { mediaPathPreviewUrl } from './mediaLoaderPreview';
import { CUSTOM_RESOLUTION_MIN, type ResolutionOptionLike } from './resolutionPicker';

export interface MediaSize {
	width: number;
	height: number;
}

export interface MediaSizeOption extends ResolutionOptionLike {
	value: string;
	description: string;
	tier: string;
	ratio: number[];
}

export const SOURCE_TIER = 'Source';
export const COMMON_TIER = 'Common sizes';
export const SCALE_TIER = 'Scale';

export const SHORT_EDGE_TARGETS = [1080, 720, 540, 480, 360];
export const SCALE_PERCENTS = [75, 50, 33, 25];

function gcd(a: number, b: number): number {
	return b === 0 ? a : gcd(b, a % b);
}

export function evenFloor(value: number): number {
	return Math.max(2, Math.floor(value / 2) * 2);
}

export function evenRound(value: number): number {
	return Math.max(2, Math.round(value / 2) * 2);
}

export function scaleSize(source: MediaSize, factor: number): MediaSize {
	return { width: evenRound(source.width * factor), height: evenRound(source.height * factor) };
}

function sourceRatio(source: MediaSize): number[] {
	const divisor = gcd(source.width, source.height) || 1;
	return [source.width / divisor, source.height / divisor];
}

export function deriveMediaSizeOptions(source: MediaSize | null | undefined): MediaSizeOption[] {
	if (!source || !(source.width > 0) || !(source.height > 0)) return [];
	const ratio = sourceRatio(source);
	const options: MediaSizeOption[] = [];
	const seen = new Set<string>();
	const add = (size: MediaSize, description: string, tier: string) => {
		if (size.width < CUSTOM_RESOLUTION_MIN || size.height < CUSTOM_RESOLUTION_MIN) return;
		const value = `${size.width}x${size.height}`;
		if (seen.has(value)) return;
		seen.add(value);
		options.push({ value, description, tier, ratio });
	};
	const original = { width: evenFloor(source.width), height: evenFloor(source.height) };
	add(original, 'Original', SOURCE_TIER);
	const shortEdge = Math.min(source.width, source.height);
	for (const edge of SHORT_EDGE_TARGETS) {
		if (edge < shortEdge) add(scaleSize(source, edge / shortEdge), `${edge}p`, COMMON_TIER);
	}
	for (const percent of SCALE_PERCENTS) {
		add(scaleSize(source, percent / 100), `${percent}%`, SCALE_TIER);
	}
	return options;
}

export function preferredMediaSize(options: MediaSizeOption[]): string | null {
	const pick =
		options.find((option) => option.tier === COMMON_TIER) ??
		options.find((option) => option.description === '50%') ??
		options[0];
	return pick?.value ?? null;
}

export function matchingSide(source: MediaSize, side: 'width' | 'height', typed: string): string {
	const text = (typed ?? '').trim();
	if (!/^\d+$/.test(text)) return '';
	const value = parseInt(text, 10);
	if (value <= 0) return '';
	const other =
		side === 'width' ? (value * source.height) / source.width : (value * source.width) / source.height;
	return String(evenRound(other));
}

export function mediaSizeFromMetadata(meta: unknown): MediaSize | null {
	if (!meta || typeof meta !== 'object') return null;
	const { width, height } = meta as { width?: unknown; height?: unknown };
	if (typeof width !== 'number' || typeof height !== 'number' || width <= 0 || height <= 0) return null;
	return { width, height };
}

export function mediaValueKey(value: unknown): string | null {
	const item = Array.isArray(value) ? value[0] : value;
	if (typeof item === 'string') return item || null;
	if (item && typeof item === 'object') {
		const record = item as Record<string, unknown>;
		const key = record.relative_path || record.path || record.url;
		return typeof key === 'string' && key ? key : null;
	}
	return null;
}

export interface MediaSizeDeps {
	metadata: MetadataSource;
	probe: (url: string, kind: 'video' | 'image') => Promise<MediaSize | null>;
}

function mediaKind(item: unknown, key: string): 'video' | 'image' {
	const declared = item && typeof item === 'object' ? (item as Record<string, unknown>).type : null;
	if (declared === 'image') return 'image';
	if (declared === 'video') return 'video';
	return /\.(png|jpe?g|webp|gif|bmp|avif)(\?|$)/i.test(key) ? 'image' : 'video';
}

export async function resolveMediaSize(value: unknown, deps: MediaSizeDeps): Promise<MediaSize | null> {
	const item = Array.isArray(value) ? value[0] : value;
	const key = mediaValueKey(item);
	if (!key) return null;
	const record = item && typeof item === 'object' ? (item as Record<string, unknown>) : null;
	const known = mediaSizeFromMetadata(record?.metadata);
	if (known) return known;
	const raw = record ? record.relative_path || record.path : item;
	if (typeof raw === 'string' && raw) {
		const fetched = mediaSizeFromMetadata(await fetchMediaMetadata(raw, deps.metadata).catch(() => null));
		if (fetched) return fetched;
	}
	const url =
		(typeof record?.url === 'string' && record.url) || (typeof raw === 'string' ? mediaPathPreviewUrl(raw) : null);
	if (!url) return null;
	try {
		return await deps.probe(url, mediaKind(item, key));
	} catch {
		return null;
	}
}

export function probeElementSize(url: string, kind: 'video' | 'image'): Promise<MediaSize | null> {
	return new Promise((resolve) => {
		if (kind === 'image') {
			const image = new Image();
			image.onload = () => resolve(mediaSizeFromMetadata({ width: image.naturalWidth, height: image.naturalHeight }));
			image.onerror = () => resolve(null);
			image.src = url;
			return;
		}
		const video = document.createElement('video');
		video.preload = 'metadata';
		video.muted = true;
		const done = (size: MediaSize | null) => {
			video.removeAttribute('src');
			video.load();
			resolve(size);
		};
		video.onloadedmetadata = () => done(mediaSizeFromMetadata({ width: video.videoWidth, height: video.videoHeight }));
		video.onerror = () => done(null);
		video.src = url;
	});
}
