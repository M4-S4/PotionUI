import type { MessageUsage } from '$lib/types/chat';

export interface UsageRow {
	key: string;
	label: string;
	value: string;
}

interface TraceUsageSource {
	prompt_tokens?: number | null;
	completion_tokens?: number | null;
	usage?: MessageUsage | null;
	cost?: number | null;
}

export function formatTokens(count: number | null | undefined): string {
	return typeof count === 'number' && Number.isFinite(count) ? count.toLocaleString('en-US') : '?';
}

export function formatCost(cost: number | null | undefined): string {
	if (typeof cost !== 'number' || !Number.isFinite(cost)) return '—';
	if (cost === 0) return '$0';
	const magnitude = Math.abs(cost);
	const digits = magnitude >= 1 ? 2 : magnitude >= 0.01 ? 3 : magnitude >= 0.0001 ? 4 : 6;
	return `$${cost.toFixed(digits)}`;
}

export function traceCost(trace: TraceUsageSource): number | null {
	return trace.cost ?? trace.usage?.cost ?? null;
}

export function traceSummary(trace: TraceUsageSource): string {
	const usage = trace.usage;
	const parts = [
		`${formatTokens(usage?.input_tokens ?? trace.prompt_tokens)}→${formatTokens(usage?.output_tokens ?? trace.completion_tokens)} tok`
	];
	if (usage?.cached_input_tokens) parts.push(`${formatTokens(usage.cached_input_tokens)} cached`);
	if (usage?.reasoning_tokens) parts.push(`${formatTokens(usage.reasoning_tokens)} reasoning`);
	const cost = traceCost(trace);
	if (cost !== null) parts.push(formatCost(cost));
	return parts.join(' · ');
}

export function usageRows(trace: TraceUsageSource): UsageRow[] {
	const usage = trace.usage;
	const tokens = (key: string, label: string, value: number | null | undefined): UsageRow[] =>
		typeof value === 'number' ? [{ key, label, value: formatTokens(value) }] : [];
	const cost = traceCost(trace);
	return [
		...tokens('input', 'Input', usage?.input_tokens ?? trace.prompt_tokens),
		...tokens('output', 'Output', usage?.output_tokens ?? trace.completion_tokens),
		...tokens('cached', 'Cached input', usage?.cached_input_tokens),
		...tokens('cache_write', 'Cache write', usage?.cache_write_tokens),
		...tokens('reasoning', 'Reasoning', usage?.reasoning_tokens),
		...(cost !== null ? [{ key: 'cost', label: 'Cost', value: formatCost(cost) }] : [])
	];
}
