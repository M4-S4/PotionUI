import { chooseBranch, parseTemplate, splice, type ConditionBlock } from './promptConditions';
import { conditionValues } from './conditionTags';
import { resolvePreviewVariables, substituteVariables, type VariableNote } from './resolvedVariables';
import {
	normalizeVariableDef,
	optionText,
	rollNumberValue,
	seededRandom,
	type VariablesMap
} from './variableDefs';

const MARK_OPEN = '⁠⁡';
const MARK_CLOSE = '⁡⁠';

export interface PreviewPiece {
	text: string;
	hit: boolean;
}

export interface PreviewRow {
	label: string;
	rolled: { name: string; value: string }[];
	branch: 'IF' | 'ELSE IF' | 'OTHERWISE' | 'NONE';
	pieces: PreviewPiece[];
}

export interface ConditionPreview {
	text: string;
	notes: VariableNote[];
	rows: PreviewRow[];
}

function readsOf(block: ConditionBlock): string[] {
	const names = new Set<string>();
	for (const branch of block.branches) for (const test of branch.cond?.tests ?? []) names.add(test.var);
	return [...names];
}

function branchLabel(block: ConditionBlock, position: number | null): PreviewRow['branch'] {
	if (position === null) return 'NONE';
	const kind = block.branches[position]?.kind;
	if (kind === 'if') return 'IF';
	return kind === 'elseif' ? 'ELSE IF' : 'OTHERWISE';
}

function splitPieces(text: string): PreviewPiece[] {
	const pieces: PreviewPiece[] = [];
	let cursor = 0;
	for (;;) {
		const open = text.indexOf(MARK_OPEN, cursor);
		if (open === -1) break;
		const close = text.indexOf(MARK_CLOSE, open + MARK_OPEN.length);
		if (close === -1) break;
		if (open > cursor) pieces.push({ text: text.slice(cursor, open), hit: false });
		pieces.push({ text: text.slice(open + MARK_OPEN.length, close), hit: true });
		cursor = close + MARK_CLOSE.length;
	}
	if (cursor < text.length) pieces.push({ text: text.slice(cursor), hit: false });
	return pieces.length > 0 ? pieces : [{ text, hit: false }];
}

function rollPerImage(name: string, variables: VariablesMap, random: () => number): string | null {
	const def = normalizeVariableDef(variables[name]);
	if (def.type === 'number') return rollNumberValue(def, random)?.value ?? null;
	if (def.type === 'choice') {
		const options = def.options.map((o) => optionText(o).trim()).filter((o) => o);
		return options.length > 0 ? options[Math.min(Math.floor(random() * options.length), options.length - 1)] : null;
	}
	return null;
}

export function previewConditions(
	text: string,
	variables: VariablesMap | undefined,
	imageCount = 4
): ConditionPreview {
	const none: ConditionPreview = { text, notes: [], rows: [] };
	if (!text.includes('<<')) return none;
	const parsed = parseTemplate(text);
	if (!parsed || parsed.blocks.length === 0) return none;
	const vars = variables ?? {};
	const settled = conditionValues(vars);
	const resolved = resolvePreviewVariables(vars);
	const notes: VariableNote[] = [];
	const noted = new Set<string>();
	const perImageNames = new Set<string>();

	const bodies = parsed.blocks.map((block) => {
		const reads = readsOf(block);
		const unsettled = reads.filter((name) => !(name in settled));
		if (unsettled.length === 0) return chooseBranch(block, settled).body;
		for (const name of unsettled) {
			const def = vars[name] === undefined ? null : normalizeVariableDef(vars[name]);
			if (def && def.type !== 'text' && def.mode === 'per-image') perImageNames.add(name);
			if (!noted.has(name) && def && def.type !== 'text') {
				noted.add(name);
				const phrase = def.mode === 'per-image' ? 're-rolls per image' : 'shuffles each generation';
				notes.push({ name, phrase: `${phrase} — the condition follows it` });
			}
		}
		return block.branches
			.map((b) => b.body.trim())
			.filter((b) => b)
			.join(' / ');
	});

	const rows: PreviewRow[] = [];
	const unsettledReads = [...new Set(parsed.blocks.flatMap(readsOf))].filter((name) => !(name in settled));
	if (unsettledReads.length > 0) {
		const perImage = unsettledReads.some((name) => perImageNames.has(name));
		const count = perImage ? imageCount : 1;
		const shuffleValues: Record<string, string> = {};
		const shuffleRandom = seededRandom(7);
		for (const name of unsettledReads) {
			if (perImageNames.has(name)) continue;
			const value = rollPerImage(name, vars, shuffleRandom);
			if (value !== null) shuffleValues[name] = value;
		}
		for (let index = 0; index < count; index++) {
			const random = seededRandom(1000 + index * 31);
			const values: Record<string, string> = { ...settled, ...shuffleValues };
			for (const name of unsettledReads) {
				if (!perImageNames.has(name)) continue;
				const value = rollPerImage(name, vars, random);
				if (value !== null) values[name] = value;
			}
			let firstBranch: PreviewRow['branch'] = 'NONE';
			const rowBodies = parsed.blocks.map((block, bi) => {
				const { position, body } = chooseBranch(block, values);
				if (bi === 0) firstBranch = branchLabel(block, position);
				return body.trim() ? `${MARK_OPEN}${body}${MARK_CLOSE}` : '';
			});
			const replacements: Record<string, string> = { ...resolved.replacements };
			for (const name of perImageNames) if (name in values) replacements[name] = values[name];
			const line = substituteVariables(splice(parsed.skeleton, rowBodies), replacements);
			rows.push({
				label: perImage ? `Image ${index + 1}` : `All ${imageCount}`,
				rolled: unsettledReads.filter((n) => n in values).map((n) => ({ name: n, value: values[n] })),
				branch: firstBranch,
				pieces: splitPieces(line)
			});
		}
	}

	return { text: splice(parsed.skeleton, bodies), notes, rows };
}
