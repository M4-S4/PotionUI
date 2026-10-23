import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import postcss from 'postcss';

const GATE = ":where(html:not([data-input='pointer']))";
const GATE_MARK = "data-input='pointer'";

const SOURCE_ROOTS = [
	fileURLToPath(new URL('../../', import.meta.url)),
	fileURLToPath(new URL('../../../../content/plugins/marketplace/system-monitor/frontend/src/', import.meta.url))
];

const TEXT_ENTRY = [
	/^input:not\(\[type='checkbox'\], \[type='radio'\], \[type='range'\]\):focus-visible$/,
	/^textarea:focus-visible$/,
	/^\[contenteditable\]:focus-visible$/,
	/^:where\( input:not\(.*\[type='checkbox'\], \[type='radio'\], \[type='range'\]\), textarea:not\(.*\) \):focus-visible$/
];

interface FocusRule {
	file: string;
	selector: string;
	gated: boolean;
	textEntry: boolean;
	rule: postcss.Rule;
}

function stylesheets(): Array<{ file: string; root: postcss.Root }> {
	const found: Array<{ file: string; root: postcss.Root }> = [];
	for (const base of SOURCE_ROOTS) {
		for (const entry of readdirSync(base, { recursive: true, encoding: 'utf8' })) {
			const file = join(base, entry);
			if (entry.endsWith('.css')) {
				found.push({ file: relative(base, file), root: postcss.parse(readFileSync(file, 'utf8')) });
			} else if (entry.endsWith('.svelte')) {
				const source = readFileSync(file, 'utf8');
				for (const block of source.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)) {
					found.push({ file: relative(base, file), root: postcss.parse(block[1]) });
				}
			}
		}
	}
	return found;
}

function hasFocusVisible(selector: string): boolean {
	return selector.replace(/:not\([^()]*:focus-visible[^()]*\)/g, '').includes(':focus-visible');
}

function drawsRing(rule: postcss.Rule): boolean {
	return rule.nodes.some((node) => {
		if (node.type === 'decl') {
			if (/^outline(-color|-style|-width)?$/.test(node.prop)) return !/^(none|0)\b/.test(node.value);
			return (node.prop === 'box-shadow' && node.value !== 'none') || node.prop.startsWith('--tw-ring');
		}
		return node.type === 'atrule' && node.name === 'apply' && /\bring-/.test(node.params);
	});
}

function subject(selector: string): string {
	const flat = selector.replace(/\s+/g, ' ').trim();
	let depth = 0;
	let start = 0;
	for (let i = 0; i < flat.length; i++) {
		const char = flat[i];
		if (char === '(' || char === '[') depth++;
		else if (char === ')' || char === ']') depth--;
		else if (depth === 0 && (char === ' ' || char === '>' || char === '+' || char === '~')) start = i + 1;
	}
	return flat.slice(start).trim();
}

function insideGate(rule: postcss.Rule): boolean {
	let parent: postcss.Container | postcss.Document | undefined = rule.parent;
	while (parent) {
		if (parent.type === 'rule' && (parent as postcss.Rule).selector.includes(GATE_MARK)) return true;
		parent = parent.parent as postcss.Container | postcss.Document | undefined;
	}
	return false;
}

function focusRules(): FocusRule[] {
	const found: FocusRule[] = [];
	for (const { file, root } of stylesheets()) {
		root.walkRules((rule) => {
			if (!drawsRing(rule)) return;
			for (const selector of rule.selectors) {
				if (!hasFocusVisible(selector)) continue;
				found.push({
					file,
					selector: selector.replace(/\s+/g, ' ').trim(),
					gated: selector.includes(GATE_MARK) || insideGate(rule),
					textEntry: TEXT_ENTRY.some((pattern) => pattern.test(subject(selector))),
					rule
				});
			}
		});
	}
	return found;
}

describe('keyboard focus rings follow the input modality', () => {
	const app = postcss.parse(readFileSync(new URL('../../app.css', import.meta.url), 'utf8'));
	const rules = focusRules();

	function appRules(selector: string): postcss.Rule[] {
		const found: postcss.Rule[] = [];
		app.walkRules((rule) => {
			if (rule.selectors.includes(selector)) found.push(rule);
		});
		return found;
	}

	function value(rule: postcss.Rule, prop: string): string | undefined {
		return rule.nodes.filter((node): node is postcss.Declaration => node.type === 'decl' && node.prop === prop).at(-1)?.value;
	}

	it('resets every focus-visible outline first so the browser default ring never shows', () => {
		const baseline = appRules(':focus-visible');
		expect(baseline).toHaveLength(1);
		expect(value(baseline[0], 'outline')).toBe('none');
	});

	it('draws the global ring in signal and only outside pointer mode', () => {
		const ring = appRules(`${GATE} :focus-visible`);
		expect(ring).toHaveLength(1);
		expect(value(ring[0], 'outline')).toBe('2px solid rgb(var(--signal))');
		expect(value(ring[0], 'outline-offset')).toBe('2px');
	});

	it('scans the app styles, the component styles and the plugin widget', () => {
		const files = new Set(rules.map((entry) => entry.file));
		for (const expected of ['app.css', 'segment-composer.css', 'chat-concept.css', 'generation-panel.css', 'PaneRow.svelte', 'SystemMonitorWidget.svelte']) {
			expect([...files].some((file) => file.endsWith(expected)), expected).toBe(true);
		}
		expect(rules.filter((entry) => entry.gated).length).toBeGreaterThan(10);
		expect(rules.filter((entry) => entry.textEntry).length).toBeGreaterThan(2);
	});

	it('gates every focus-visible ring except text entry', () => {
		const ungated = rules.filter((entry) => !entry.gated && !entry.textEntry).map((entry) => `${entry.file}: ${entry.selector}`);
		expect(ungated).toEqual([]);
	});

	it('never paints a focus-visible ring in the white accent colour', () => {
		const accent = rules.filter((entry) => entry.rule.toString().includes('var(--accent')).map((entry) => `${entry.file}: ${entry.selector}`);
		expect(accent).toEqual([]);
	});

	it('keeps text entry rings off checkboxes, radios and ranges', () => {
		const loose = rules.filter((entry) => !entry.gated && /(^|\s)input:focus-visible$/.test(subject(entry.selector)));
		expect(loose.map((entry) => `${entry.file}: ${entry.selector}`)).toEqual([]);
	});
});
