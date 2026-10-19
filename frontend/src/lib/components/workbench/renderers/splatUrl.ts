export interface SplatMetadataLike {
	splat_count?: number | null;
	filename?: string | null;
	format?: string | null;
}

export interface SplatFileLike {
	current_splat?: string | null;
	splat_metadata?: SplatMetadataLike | null;
	url?: string | null;
	originalUrl?: string | null;
	file_path?: string | null;
	splat_count?: number | null;
	splat_name?: string | null;
	splat_format?: string | null;
}

export type SplatContainerFormat = 'ply' | 'spz' | 'splat';

const CONTAINER_FORMATS: readonly string[] = ['ply', 'spz', 'splat'];

export function resolveSplatUrl(file: SplatFileLike | null | undefined): string | null {
	if (!file) return null;
	return file.current_splat || file.originalUrl || file.url || null;
}

export interface ResolvedSplatMetadata {
	splatCount: number | null;
	filename: string | null;
	format: string | null;
}

export function resolveSplatMetadata(file: SplatFileLike | null | undefined): ResolvedSplatMetadata {
	const meta = file?.splat_metadata;
	return {
		splatCount: meta?.splat_count ?? file?.splat_count ?? null,
		filename: meta?.filename ?? file?.splat_name ?? null,
		format: meta?.format ?? file?.splat_format ?? null
	};
}

function extensionOf(path: string | null | undefined): string | null {
	if (!path) return null;
	const basename = path.split(/[?#]/)[0].split('/').pop() ?? '';
	if (!basename.includes('.')) return null;
	return (basename.split('.').pop() ?? '').toLowerCase() || null;
}

export function resolveSplatFormat(file: SplatFileLike | null | undefined): SplatContainerFormat | null {
	const declared = resolveSplatMetadata(file).format?.toLowerCase();
	if (declared && CONTAINER_FORMATS.includes(declared)) return declared as SplatContainerFormat;
	const extension =
		extensionOf(resolveSplatUrl(file)) ?? extensionOf(file?.file_path) ?? extensionOf(resolveSplatMetadata(file).filename);
	return extension && CONTAINER_FORMATS.includes(extension) ? (extension as SplatContainerFormat) : null;
}
