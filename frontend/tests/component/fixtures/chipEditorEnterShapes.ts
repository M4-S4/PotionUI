import type { ChipData } from '$lib/types/segments';

export interface EnterShape {
	name: string;
	html: string;
	value: string;
	lines: number;
}

const CHIP =
	'<span data-chip-id="c1" contenteditable="false" style="user-select: none; display: inline;">[chip]</span>';

export const ENTER_SHAPE_CHIPS: Record<string, ChipData> = {
	c1: {
		id: 'c1',
		categoryPath: 'cat',
		valueId: 'val-1',
		label: 'Cat',
		value: 'cat',
		allValues: [],
		shuffle: false,
		autoRegen: false
	}
};

export const ENTER_SHAPES: EnterShape[] = [
	{
		name: 'three typed lines',
		html: 'line one<div>line two</div><div>line three</div>',
		value: 'line one\nline two\nline three',
		lines: 3
	},
	{ name: 'text then a typed line', html: 'a<div>b</div>', value: 'a\nb', lines: 2 },
	{ name: 'a blank line between two lines', html: 'a<div><br></div><div>b</div>', value: 'a\n\nb', lines: 3 },
	{ name: 'Enter at the end of a line', html: 'a<div><br></div>', value: 'a\n', lines: 2 },
	{ name: 'Enter first, then a line', html: '<div><br></div><div>b</div>', value: '\nb', lines: 2 },
	{ name: 'Enter twice in an empty editor', html: '<div><br></div><div><br></div>', value: '\n', lines: 2 },
	{
		name: 'a blank line in the middle',
		html: 'line one<div>line two</div><div><br></div><div>line four</div>',
		value: 'line one\nline two\n\nline four',
		lines: 4
	},
	{ name: 'Enter after a chip', html: `${CHIP}<div>after</div>`, value: '#cat\nafter', lines: 2 },
	{
		name: 'Enter after text, a chip and more text',
		html: `before ${CHIP} tail<div>next</div>`,
		value: 'before #cat tail\nnext',
		lines: 2
	},
	{
		name: 'Enter at the end of a line that ends in a chip',
		html: `before ${CHIP}<div><br></div>`,
		value: 'before #cat\n',
		lines: 2
	},
	{ name: 'Enter in the middle of a word run', html: 'alpha<div> beta</div>', value: 'alpha\n beta', lines: 2 },
	{
		name: 'Enter after a literal newline',
		html: 'alpha\nbeta<div>gamma</div>',
		value: 'alpha\nbeta\ngamma',
		lines: 3
	},
	{
		name: 'Enter after a trailing literal newline',
		html: 'alpha\n<div>gamma</div>',
		value: 'alpha\ngamma',
		lines: 2
	},
	{ name: 'Shift+Enter', html: 'alpha\nbeta', value: 'alpha\nbeta', lines: 2 },
	{
		name: 'Shift+Enter then Enter',
		html: 'alpha\n\n<div>beta</div>',
		value: 'alpha\n\nbeta',
		lines: 3
	},
	{
		name: 'a blank line between block lines',
		html: '<div>a</div><div><br></div><div>b</div>',
		value: 'a\n\nb',
		lines: 3
	},
	{ name: 'a line break element between text', html: 'a<br>b', value: 'a\nb', lines: 2 },
	{ name: 'a line break element before a block', html: 'alpha<br><div>beta</div>', value: 'alpha\nbeta', lines: 2 },
	{ name: 'a block followed by text', html: '<div>line</div>tail', value: 'line\ntail', lines: 2 },
	{ name: 'a block followed by a chip', html: `<div>line</div>${CHIP}`, value: 'line\n#cat', lines: 2 },
	{ name: 'only block lines', html: '<div>one</div><div>two</div><div>three</div>', value: 'one\ntwo\nthree', lines: 3 },
	{ name: 'a lone placeholder line in a block', html: '<div><br></div>', value: '', lines: 1 },
	{ name: 'a lone placeholder line break', html: '<br>', value: '', lines: 1 }
];
