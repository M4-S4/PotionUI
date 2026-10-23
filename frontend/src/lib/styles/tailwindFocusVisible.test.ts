import { describe, expect, it } from 'vitest';
import postcss from 'postcss';
import tailwindcss from 'tailwindcss';

const GATE = ":where(html:not([data-input='pointer']) *)";

const CLASSES = [
	'hover:bg-surface-3',
	'focus:ring-1',
	'focus-visible:ring-2',
	'focus-visible:outline-none',
	'focus-visible:hover:underline',
	'md:focus-visible:underline',
	'group-hover:focus-visible:underline',
	'group-focus-visible:opacity-100',
	'peer-focus-visible:ring-2',
	'peer-hover:focus-visible:underline',
	'active:scale-95',
	'disabled:opacity-50'
];

const CONFIG_URL = new URL('../../../tailwind.config.js', import.meta.url).href;

async function compile(): Promise<postcss.Root> {
	const { default: config } = await import(CONFIG_URL);
	const options = {
		...config,
		content: [{ raw: `<div class="${CLASSES.join(' ')}"></div>` }],
		corePlugins: { preflight: false }
	} as unknown as Parameters<typeof tailwindcss>[0];
	const result = await postcss([tailwindcss(options)]).process('@tailwind utilities;', { from: undefined });
	return postcss.parse(result.css);
}

function selectorsMentioning(root: postcss.Root, fragment: string): string[] {
	const found: string[] = [];
	root.walkRules((rule) => {
		if (rule.selector.includes(fragment)) found.push(rule.selector);
	});
	return found;
}

const GATED: Array<[string, string]> = [
	['.focus-visible\\:ring-2', `.focus-visible\\:ring-2:focus-visible${GATE}`],
	['.focus-visible\\:outline-none', `.focus-visible\\:outline-none:focus-visible${GATE}`],
	['.focus-visible\\:hover\\:underline', `.focus-visible\\:hover\\:underline:hover:focus-visible${GATE}`],
	['.md\\:focus-visible\\:underline', `.md\\:focus-visible\\:underline:focus-visible${GATE}`],
	['.group-hover\\:focus-visible\\:underline', `.group:hover .group-hover\\:focus-visible\\:underline:focus-visible${GATE}`],
	['.group-focus-visible\\:opacity-100', `.group:focus-visible${GATE} .group-focus-visible\\:opacity-100`],
	['.peer-focus-visible\\:ring-2', `.peer:focus-visible${GATE} ~ .peer-focus-visible\\:ring-2`],
	['.peer-hover\\:focus-visible\\:underline', `.peer:hover ~ .peer-hover\\:focus-visible\\:underline:focus-visible${GATE}`]
];

describe('tailwind focus-visible variants', () => {
	it.each(GATED)('gates %s on the element that takes focus', async (fragment, expected) => {
		expect(selectorsMentioning(await compile(), fragment)).toEqual([expected]);
	});

	it('leaves the other state variants ungated', async () => {
		const root = await compile();
		for (const fragment of ['.hover\\:bg-surface-3', '.focus\\:ring-1', '.active\\:scale-95', '.disabled\\:opacity-50']) {
			const selectors = selectorsMentioning(root, fragment);
			expect(selectors, fragment).toHaveLength(1);
			expect(selectors[0], fragment).not.toContain('data-input');
		}
	});

	it('keeps focus-visible between focus and active in the cascade', async () => {
		const order = (await compile()).nodes.flatMap((node) => (node.type === 'rule' ? [node.selector] : []));
		const at = (fragment: string) => order.findIndex((selector) => selector.includes(fragment));
		expect(at('.focus\\:ring-1')).toBeLessThan(at('.focus-visible\\:ring-2'));
		expect(at('.focus-visible\\:ring-2')).toBeLessThan(at('.active\\:scale-95'));
	});
});
