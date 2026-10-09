import { formatBytes as formatByteCount } from '$lib/utils/format';
import type { LimitKindDescriptor, LimitValueType } from './types';

const MB = 1024 ** 2;
const GB = 1024 ** 3;
const TB = 1024 ** 4;

export function trimNumber(value: number, digits = 1): string {
	return Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

export function formatBytes(bytes: number): string {
	if (bytes >= TB) return `${trimNumber(bytes / TB, 2)} TB`;
	if (bytes >= GB / 10) return `${trimNumber(bytes / GB, 2)} GB`;
	if (bytes >= MB) return `${trimNumber(bytes / MB)} MB`;
	return `${Math.round(bytes)} B`;
}

export function formatUsd(value: number): string {
	return Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`;
}

export function formatValue(valueType: LimitValueType, value: number): string {
	if (valueType === 'bytes') return formatBytes(value);
	if (valueType === 'usd') return formatUsd(value);
	return String(Math.round(value));
}

export function formatLimitValue(
	kind: Pick<LimitKindDescriptor, 'value_type'> & { per_item?: boolean },
	value: number | null
): string {
	if (value === null) return 'no limit';
	if (kind.per_item && kind.value_type === 'bytes') return formatByteCount(value, 1);
	return formatValue(kind.value_type, value);
}

export function windowSuffix(window: LimitKindDescriptor['window']): string {
	if (window === 'day') return '/ day';
	if (window === 'month') return '/ mo';
	return '';
}

export function shortLabel(kind: Pick<LimitKindDescriptor, 'label'>): string {
	return kind.label.replace(/ (space|spend per month|spend|per day|per month)$/i, '');
}

export function formatLimitChip(kind: LimitKindDescriptor, value: number): string {
	const suffix = windowSuffix(kind.window);
	return `${shortLabel(kind)} ${formatLimitValue(kind, value)}${suffix ? ` ${suffix}` : ''}`;
}

export function kindIcon(kind: Pick<LimitKindDescriptor, 'icon' | 'value_type' | 'plugin'>): string {
	if (kind.icon) return kind.icon;
	if (kind.plugin) return 'extension';
	if (kind.value_type === 'bytes') return 'database';
	if (kind.value_type === 'usd') return 'calendar';
	return 'bolt';
}

export function kindUnitLabel(kind: Pick<LimitKindDescriptor, 'value_type' | 'unit' | 'window'>): string {
	if (kind.value_type === 'bytes') return '';
	return kind.unit;
}

export function percent(used: number, limit: number | null): number | null {
	if (limit === null) return null;
	if (limit <= 0) return 100;
	return Math.min(100, Math.round((used / limit) * 100));
}
