export interface ModelFileWorstLayer {
	key: string;
	rel_l2?: number | null;
	cosine?: number | null;
}

export interface ModelFileSource {
	name: string;
	sha256?: string | null;
	strength?: number | null;
}

export interface ModelFileSummary {
	layers_quantized?: number | null;
	layers_kept?: number | null;
	worst_layers?: ModelFileWorstLayer[] | null;
	energy_kept?: number | null;
	rank?: number | null;
	sources?: ModelFileSource[] | null;
	licence_note?: string | null;
}

export interface ModelFileData {
	index?: number;
	kind?: 'model' | 'lora' | string;
	model_type?: string | null;
	file_name?: string | null;
	size_bytes?: number | null;
	source_size_bytes?: number | null;
	format?: string | null;
	summary?: ModelFileSummary | null;
	model_id?: string | null;
	indexed?: boolean;
	index_warning?: string | null;
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];
const STEP = 1024;

function unitIndex(bytes: number): number {
	if (!(bytes > 0)) return 0;
	return Math.min(Math.floor(Math.log(bytes) / Math.log(STEP)), UNITS.length - 1);
}

function scaled(bytes: number, unit: number): string {
	const value = bytes / Math.pow(STEP, unit);
	return unit === 0 ? String(Math.round(value)) : value.toFixed(1);
}

export function formatModelFileSizes(
	before: number | null | undefined,
	after: number | null | undefined
): string {
	const hasAfter = typeof after === 'number' && after > 0;
	const hasBefore = typeof before === 'number' && before > 0;
	if (!hasAfter && !hasBefore) return '';
	if (!hasBefore) {
		const unit = unitIndex(after as number);
		return `${scaled(after as number, unit)} ${UNITS[unit]}`;
	}
	if (!hasAfter) {
		const unit = unitIndex(before as number);
		return `${scaled(before as number, unit)} ${UNITS[unit]}`;
	}
	const unit = unitIndex(Math.max(before as number, after as number));
	return `${scaled(before as number, unit)} → ${scaled(after as number, unit)} ${UNITS[unit]}`;
}

export function sizeReductionPercent(
	before: number | null | undefined,
	after: number | null | undefined
): number | null {
	if (typeof before !== 'number' || typeof after !== 'number') return null;
	if (!(before > 0) || !(after > 0)) return null;
	return Math.round((1 - after / before) * 100);
}

export function formatEnergyKept(value: number | null | undefined): string | null {
	if (typeof value !== 'number' || !Number.isFinite(value)) return null;
	const percent = value <= 1 ? value * 100 : value;
	return `${percent.toFixed(1)}%`;
}

export function isLoraFile(data: ModelFileData): boolean {
	return data.kind === 'lora';
}

export function modelFileHref(data: ModelFileData): string | null {
	return data.model_id ? `/models/${encodeURIComponent(data.model_id)}` : null;
}

export function modelFileTypeLabel(data: ModelFileData): string {
	const raw = data.model_type || (isLoraFile(data) ? 'lora' : 'model');
	return raw.replace(/_/g, ' ');
}

export function hasQualitySummary(summary: ModelFileSummary | null | undefined): boolean {
	if (!summary) return false;
	return (
		typeof summary.layers_quantized === 'number' ||
		typeof summary.layers_kept === 'number' ||
		typeof summary.energy_kept === 'number' ||
		typeof summary.rank === 'number'
	);
}

export function splitFileName(name: string): { stem: string; ext: string } {
	const dot = name.lastIndexOf('.');
	if (dot <= 0) return { stem: name, ext: '' };
	return { stem: name.slice(0, dot), ext: name.slice(dot) };
}

export function normalizeLabel(value: string): string {
	return value.toLowerCase().replace(/[\s_-]+/g, ' ').trim();
}
