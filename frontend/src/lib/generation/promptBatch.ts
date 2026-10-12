import type { PromptTabData, Tab } from '$lib/types/tabs';
import { resolvePromptSegments } from '$lib/utils/promptSegments';
import { ensureSegmentList } from '$lib/utils/richSegments';

export const MAX_BATCH_IMAGES = 20;

export type PromptFields = Pick<
	Tab,
	'prompt' | 'negativePrompt' | 'promptSegments' | 'negativePromptSegments' | 'promptTabs' | 'activePromptTab'
>;

export type PromptListPatch = Pick<
	Tab,
	'prompt' | 'negativePrompt' | 'promptSegments' | 'negativePromptSegments' | 'promptTabs' | 'activePromptTab'
>;

export function canBatchPrompts(source: { supports_prompt_batch?: boolean } | null | undefined): boolean {
	return source?.supports_prompt_batch === true;
}

export function quantityOf(formData: Record<string, unknown> | null | undefined): number {
	const value = Math.floor(Number(formData?.quantity ?? 1));
	return Number.isFinite(value) && value >= 1 ? value : 1;
}

export function emptyPrompt(): PromptTabData {
	return {
		promptSegments: ensureSegmentList([]),
		negativePromptSegments: ensureSegmentList([]),
		prompt: '',
		negativePrompt: ''
	};
}

function workingPrompt(tab: PromptFields): PromptTabData {
	return {
		promptSegments: tab.promptSegments || [],
		negativePromptSegments: tab.negativePromptSegments || [],
		prompt: tab.prompt || '',
		negativePrompt: tab.negativePrompt || ''
	};
}

export function promptCount(tab: Pick<Tab, 'promptTabs'>): number {
	const tabs = tab.promptTabs;
	return tabs && tabs.length > 1 ? tabs.length : 1;
}

export function activePromptIndex(tab: Pick<Tab, 'promptTabs' | 'activePromptTab'>): number {
	const count = promptCount(tab);
	const raw = Math.floor(Number(tab.activePromptTab ?? 0));
	if (!Number.isFinite(raw)) return 0;
	return Math.min(Math.max(raw, 0), count - 1);
}

export function promptList(tab: PromptFields): PromptTabData[] {
	if (promptCount(tab) < 2) return [workingPrompt(tab)];
	return [...tab.promptTabs!];
}

function buildPatch(list: PromptTabData[], active: number): PromptListPatch {
	if (list.length < 2) {
		const only = list[0] ?? emptyPrompt();
		return {
			promptTabs: undefined,
			activePromptTab: undefined,
			promptSegments: only.promptSegments,
			negativePromptSegments: only.negativePromptSegments,
			prompt: only.prompt,
			negativePrompt: only.negativePrompt
		};
	}
	const index = Math.min(Math.max(active, 0), list.length - 1);
	const current = list[index];
	return {
		promptTabs: list,
		activePromptTab: index,
		promptSegments: current.promptSegments,
		negativePromptSegments: current.negativePromptSegments,
		prompt: current.prompt,
		negativePrompt: current.negativePrompt
	};
}

export function selectPrompt(tab: PromptFields, index: number): PromptListPatch {
	return buildPatch(promptList(tab), index);
}

export function addPrompt(tab: PromptFields): PromptListPatch {
	const list = [...promptList(tab), emptyPrompt()];
	return buildPatch(list, list.length - 1);
}

export function duplicatePrompt(tab: PromptFields, index: number): PromptListPatch {
	const list = promptList(tab);
	const source = list[index];
	if (!source) return buildPatch(list, activePromptIndex(tab));
	const copy: PromptTabData = structuredClone(source);
	if (copy.name?.trim()) copy.name = `${copy.name.trim()} copy`;
	else delete copy.name;
	list.splice(index + 1, 0, copy);
	return buildPatch(list, index + 1);
}

export function deletePrompt(tab: PromptFields, index: number): PromptListPatch | null {
	const list = promptList(tab);
	if (list.length < 2 || index < 0 || index >= list.length) return null;
	const active = activePromptIndex(tab);
	list.splice(index, 1);
	const next = index < active ? active - 1 : Math.min(active, list.length - 1);
	return buildPatch(list, next);
}

export function renamePrompt(tab: PromptFields, index: number, name: string): PromptListPatch {
	const list = promptList(tab);
	if (list[index]) {
		const trimmed = name.trim();
		const renamed = { ...list[index] };
		if (trimmed) renamed.name = trimmed;
		else delete renamed.name;
		list[index] = renamed;
	}
	return buildPatch(list, activePromptIndex(tab));
}

const PROMPT_FIELD_KEYS = ['prompt', 'negativePrompt', 'promptSegments', 'negativePromptSegments'] as const;

export function mirrorActivePrompt(next: PromptFields, updates: Partial<PromptFields>): Partial<PromptFields> | null {
	if (!next.promptTabs || next.promptTabs.length < 2) return null;
	if ('promptTabs' in updates || 'activePromptTab' in updates) return null;
	if (!PROMPT_FIELD_KEYS.some((key) => key in updates)) return null;
	const active = activePromptIndex(next);
	const tabs = [...next.promptTabs];
	tabs[active] = { ...tabs[active], ...workingPrompt(next) };
	return { promptTabs: tabs };
}

export function promptTitle(entry: PromptTabData, index: number): string {
	const name = entry.name?.trim();
	if (name) return name;
	const firstLine = (entry.prompt || '')
		.split('\n')
		.map((line) => line.trim())
		.find(Boolean);
	return firstLine || `Prompt ${index + 1}`;
}

export function hasPromptText(entry: PromptTabData): boolean {
	const fromSegments = entry.promptSegments?.length ? resolvePromptSegments(entry.promptSegments).trim() : '';
	return fromSegments.length > 0 || (entry.prompt || '').trim().length > 0;
}

export function firstEmptyPrompt(list: PromptTabData[]): number {
	return list.findIndex((entry) => !hasPromptText(entry));
}

export interface BatchSummary {
	prompts: number;
	sent: number;
	perPrompt: number;
	total: number;
	overCap: boolean;
	unsupported: boolean;
	maxPrompts: number;
}

export function batchSummary(input: { prompts: number; perPrompt: number; canBatch: boolean }): BatchSummary {
	const prompts = Math.max(1, Math.floor(input.prompts));
	const perPrompt = Math.max(1, Math.floor(input.perPrompt));
	const unsupported = prompts > 1 && !input.canBatch;
	const sent = unsupported ? 1 : prompts;
	const total = sent * perPrompt;
	return {
		prompts,
		sent,
		perPrompt,
		total,
		overCap: prompts > 1 && !unsupported && total > MAX_BATCH_IMAGES,
		unsupported,
		maxPrompts: Math.max(1, Math.floor(MAX_BATCH_IMAGES / perPrompt))
	};
}

function plural(count: number, word: string): string {
	return `${count} ${word}${count === 1 ? '' : 's'}`;
}

export function batchLine(summary: BatchSummary): { text: string; generateLabel: string; over: boolean } | null {
	if (summary.prompts < 2) return null;
	const lead = summary.unsupported ? `1 of ${summary.prompts} prompts` : plural(summary.prompts, 'prompt');
	return {
		text: `${lead} × ${plural(summary.perPrompt, 'image')} = ${plural(summary.total, 'image')}`,
		generateLabel: `Generate ${plural(summary.total, 'image')}`,
		over: summary.overCap
	};
}

export const BATCH_CAP_MESSAGE = `Up to ${MAX_BATCH_IMAGES} images per run.`;

export function batchPromptFields(
	entries: Array<{ prompt_index?: unknown; prompt?: unknown } | null | undefined>,
	index: number
): { promptIndex: number; promptText?: string; promptImage: number } | null {
	const promptIndex = entries[index]?.prompt_index;
	if (typeof promptIndex !== 'number' || !Number.isInteger(promptIndex) || promptIndex < 0) return null;
	let promptImage = 1;
	for (let i = 0; i < index; i++) {
		if (entries[i]?.prompt_index === promptIndex) promptImage += 1;
	}
	const text = entries[index]?.prompt;
	return { promptIndex, ...(typeof text === 'string' ? { promptText: text } : {}), promptImage };
}

export function batchTileLabel(batch: { promptIndex: number; promptImage: number } | null): string | null {
	return batch ? `${batch.promptIndex + 1}.${batch.promptImage}` : null;
}
