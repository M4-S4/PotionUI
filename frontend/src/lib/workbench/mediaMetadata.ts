import { formatBytes } from '$lib/utils/format';

export interface MediaMetadata {
	width: number;
	height: number;
	fileSize: string;
	format: string;
}

export interface MediaFileMetadata {
	width?: number | null;
	height?: number | null;
	file_size?: number | null;
	mime_type?: string | null;
	file_path?: string | null;
}

interface GenerationSource {
	video_metadata?: { resolution?: number[] | null } | null;
}

export const UNKNOWN_SIZE = 'Unknown';

const VIDEO_EXTENSIONS: Array<[string, string]> = [
	['.webm', 'WEBM'],
	['.avi', 'AVI'],
	['.mov', 'MOV']
];

function resolutionOf(source: object | null | undefined): [number, number] | null {
	const resolution = (source as { resolution?: unknown } | null | undefined)?.resolution;
	if (!Array.isArray(resolution) || resolution.length < 2) return null;
	return [resolution[0], resolution[1]];
}

function sizeLabel(bytes: number | null | undefined): string {
	return bytes ? formatBytes(bytes) : UNKNOWN_SIZE;
}

function videoFormat(url: string | null | undefined): string {
	if (!url) return 'MP4';
	return VIDEO_EXTENSIONS.find(([extension]) => url.includes(extension))?.[1] ?? 'MP4';
}

function imageFormat(file: MediaFileMetadata): string {
	if (file.mime_type) {
		const subtype = file.mime_type.split('/')[1];
		return subtype ? subtype.toUpperCase() : 'PNG';
	}
	if (file.file_path) {
		const extension = file.file_path.split('.').pop();
		return extension ? extension.toUpperCase() : 'PNG';
	}
	return 'PNG';
}

export function deriveMediaMetadata(
	kind: 'image' | 'video',
	fileMetadata: MediaFileMetadata | null | undefined,
	item: object | null | undefined,
	generation: GenerationSource | null | undefined,
	url?: string | null
): MediaMetadata | null {
	const hasFileDimensions = !!(fileMetadata?.width && fileMetadata?.height);
	const itemResolution = resolutionOf(item);

	if (kind === 'video') {
		let width = 0;
		let height = 0;
		if (fileMetadata?.width && fileMetadata?.height) {
			width = fileMetadata.width;
			height = fileMetadata.height;
		} else if (itemResolution) {
			[width, height] = itemResolution;
		} else {
			const fallback = generation?.video_metadata?.resolution;
			if (fallback) {
				width = fallback[0];
				height = fallback[1];
			}
		}
		return { width, height, fileSize: sizeLabel(fileMetadata?.file_size), format: videoFormat(url) };
	}

	if (hasFileDimensions && fileMetadata) {
		return {
			width: fileMetadata.width as number,
			height: fileMetadata.height as number,
			fileSize: sizeLabel(fileMetadata.file_size),
			format: imageFormat(fileMetadata)
		};
	}
	if (itemResolution) {
		return { width: itemResolution[0], height: itemResolution[1], fileSize: UNKNOWN_SIZE, format: 'PNG' };
	}
	return null;
}

export function shouldProbeSize(input: {
	kind: 'image' | 'video';
	filesReady: boolean;
	fileMetadata: MediaFileMetadata | null | undefined;
	metadata: MediaMetadata | null | undefined;
	url: string | null | undefined;
}): boolean {
	const { kind, filesReady, fileMetadata, metadata, url } = input;
	if (!filesReady || !metadata || metadata.fileSize !== UNKNOWN_SIZE || !url) return false;
	if (url.startsWith('blob:') || url.startsWith('data:')) return false;
	if (kind === 'image' && fileMetadata?.width && fileMetadata?.height) return false;
	return true;
}

const probes = new Map<string, Promise<number | null>>();

export function probeContentLength(url: string): Promise<number | null> {
	let probe = probes.get(url);
	if (!probe) {
		probe = fetch(url, { method: 'HEAD' })
			.then((response) => {
				const length = parseInt(response.headers.get('content-length') ?? '', 10);
				return Number.isFinite(length) && length > 0 ? length : null;
			})
			.catch(() => null);
		probes.set(url, probe);
	}
	return probe;
}

export function clearProbeCache(): void {
	probes.clear();
}

export function withProbedSize(metadata: MediaMetadata, bytes: number): MediaMetadata {
	return { ...metadata, fileSize: formatBytes(bytes) };
}
