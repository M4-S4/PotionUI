// @vitest-environment jsdom
import { readFileSync } from 'node:fs';
import postcss, { type ChildNode, type Container, type Declaration, type Document, type Rule } from 'postcss';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { mountComponent } from './helpers/mount';

const { default: InlineChipEditor } = await import('$lib/components/InlineChipEditor.svelte');

const variables: VariablesMap = {
	age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 18 }
};

const CHAIN = 'portrait, <<if $age > 65>>grey hair<<else if $age > 40>>brown hair<<else>>dark<<end>>, soft light';
const CHOICE = 'portrait, {glowing|shimmering}, <<if $age > 65>>grey hair<<end>>';

function selectorsOf(rule: Rule): string[] {
	const parent = rule.parent;
	if (parent?.type !== 'rule') return rule.selectors;
	return selectorsOf(parent as Rule).flatMap((outer) =>
		rule.selectors.map((own) => (own.includes('&') ? own.replaceAll('&', outer) : `${outer} ${own}`))
	);
}

function insideAtRule(node: ChildNode): boolean {
	let ancestor: Container | Document | undefined = node.parent;
	while (ancestor) {
		if (ancestor.type === 'atrule') return true;
		ancestor = ancestor.parent;
	}
	return false;
}

function flatChipStylesheet(): string {
	const root = postcss.parse(readFileSync('src/lib/styles/segment-composer.css', 'utf8'));
	const out: string[] = [];
	root.walkRules((rule) => {
		if (insideAtRule(rule)) return;
		const declarations = (rule.nodes ?? [])
			.filter((node): node is Declaration => node.type === 'decl')
			.map((node) => `${node.prop}: ${node.value}${node.important ? ' !important' : ''};`)
			.join(' ');
		if (!declarations) return;
		for (const selector of selectorsOf(rule)) {
			if (selector.includes('chip')) out.push(`${selector} { ${declarations} }`);
		}
	});
	return out.join('\n');
}

let style: HTMLStyleElement;

beforeEach(() => {
	style = document.createElement('style');
	style.textContent = flatChipStylesheet();
	document.head.appendChild(style);
});

afterEach(() => {
	style.remove();
});

function mountSegment(value = CHAIN) {
	const composer = document.body.appendChild(document.createElement('div'));
	composer.className = 'segment-composer';
	const content = composer.appendChild(document.createElement('div'));
	content.className = 'segment-content';
	mountComponent(InlineChipEditor as never, { value, chips: {}, variant: 'segment-composer', borderless: true, variables }, { target: content });
	return composer;
}

describe('segment composer condition chip layout', () => {
	it('is allowed to grow past the 260px cap every other composer chip has', () => {
		const chip = mountSegment().querySelector<HTMLElement>('.condition-chip')!;
		expect(getComputedStyle(chip).maxWidth).toBe('min(100%, 560px)');
	});

	it('lets the head shrink to an ellipsis so it never runs under the settings button', () => {
		const head = mountSegment().querySelector<HTMLElement>('.condition-chip .chip-context')!;
		const computed = getComputedStyle(head);
		expect(parseFloat(computed.minWidth)).toBe(0);
		expect(computed.overflow).toBe('hidden');
		expect(computed.getPropertyValue('text-overflow')).toBe('ellipsis');
	});

	it('keeps the +N badge at its own width instead of letting it spill over the settings button', () => {
		const badge = mountSegment().querySelector<HTMLElement>('.condition-chip .chip-state')!;
		expect(badge.textContent?.trim()).toBe('+1');
		const computed = getComputedStyle(badge);
		expect(computed.flexShrink).toBe('0');
		expect(computed.flexGrow).toBe('0');
	});
});

describe('segment composer choice group chip mark', () => {
	it('sizes to its braces instead of the fixed 16px box they overflow', () => {
		const mark = mountSegment(CHOICE).querySelector<HTMLElement>('.choice-chip .chip-mark')!;
		expect(mark.textContent).toBe('{ }');
		const computed = getComputedStyle(mark);
		expect(computed.width).toBe('auto');
		expect(computed.minWidth).toBe('16px');
		expect(computed.paddingLeft).toBe('2px');
		expect(computed.paddingRight).toBe('2px');
	});

	it('leaves the condition chip mark at its 16px box', () => {
		const mark = mountSegment(CHOICE).querySelector<HTMLElement>('.condition-chip .chip-mark')!;
		expect(mark.textContent).toBe('if');
		const computed = getComputedStyle(mark);
		expect(computed.width).toBe('16px');
		expect(computed.minWidth).not.toBe('16px');
	});
});
