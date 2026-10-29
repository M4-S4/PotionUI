import { describe, it, expect, vi } from 'vitest';
import { flushSync } from 'svelte';
import { mountComponent } from './helpers/mount';

const { default: SegmentedField } = await import('$lib/components/form-fields/SegmentedField.svelte');
const { default: AnchorField } = await import('$lib/components/form-fields/AnchorField.svelte');
const { default: ColorField } = await import('$lib/components/form-fields/ColorField.svelte');
const { default: NumberBoxField } = await import('$lib/components/form-fields/NumberBoxField.svelte');
const { default: SliderField } = await import('$lib/components/form-fields/SliderField.svelte');

function type(input: HTMLInputElement, text: string) {
	input.focus();
	input.value = text;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	flushSync();
}

function blur(input: HTMLInputElement) {
	input.blur();
	flushSync();
}

describe('SegmentedField', () => {
	const options = [
		{ label: 'Auto', value: 'auto' },
		{ label: 'Color', value: 'color' }
	];

	it('draws one button per option and marks the current one', () => {
		const { target } = mountComponent(SegmentedField, {
			name: 'key_mode',
			config: { title: 'Key', options },
			value: 'color',
			onChange: vi.fn()
		});

		const buttons = Array.from(target.querySelectorAll('button'));
		expect(buttons.map((button) => button.textContent?.trim())).toEqual(['Auto', 'Color']);
		expect(buttons.map((button) => button.getAttribute('aria-pressed'))).toEqual(['false', 'true']);
	});

	it('reports the picked option under the field name', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(SegmentedField, { name: 'key_mode', config: { options }, value: 'auto', onChange });

		(target.querySelectorAll('button')[1] as HTMLButtonElement).click();

		expect(onChange).toHaveBeenCalledWith('key_mode', 'color');
	});

	it('hands back a numeric option as the number it was declared as', () => {
		const onChange = vi.fn();
		const degrees = [90, 180, 270].map((value) => ({ label: `${value}°`, value }));
		const { target } = mountComponent(SegmentedField, { name: 'degrees', config: { options: degrees }, value: 90, onChange });

		(target.querySelectorAll('button')[2] as HTMLButtonElement).click();

		expect(onChange).toHaveBeenCalledWith('degrees', 270);
	});

	it('follows the value when the form changes it', () => {
		const view = mountComponent(SegmentedField, { name: 'k', config: { options }, value: 'auto', onChange: vi.fn() });
		view.rerender({ value: 'color' });
		const pressed = view.target.querySelector('[aria-pressed="true"]');
		expect(pressed?.textContent?.trim()).toBe('Color');
	});

	it('shows the field title as its group name', () => {
		const { target } = mountComponent(SegmentedField, { name: 'k', config: { title: 'Overlap', options }, value: 'auto', onChange: vi.fn() });
		expect(target.textContent).toContain('Overlap');
		expect(target.querySelector('[role="group"]')?.getAttribute('aria-label')).toBe('Overlap');
	});
});

describe('AnchorField', () => {
	it('draws nine cells and marks the current anchor', () => {
		const { target } = mountComponent(AnchorField, { name: 'anchor', config: { title: 'Anchor' }, value: 'top_left', onChange: vi.fn() });

		const cells = Array.from(target.querySelectorAll('button[data-anchor]'));
		expect(cells).toHaveLength(9);
		expect(cells.map((cell) => cell.getAttribute('data-anchor'))).toEqual([
			'top_left',
			'top',
			'top_right',
			'left',
			'center',
			'right',
			'bottom_left',
			'bottom',
			'bottom_right'
		]);
		expect(target.querySelector('[aria-pressed="true"]')?.getAttribute('data-anchor')).toBe('top_left');
		expect(target.querySelector('[data-anchor-readout]')?.textContent).toBe('Top left');
	});

	it('falls back to the centre for a value it does not know', () => {
		const { target } = mountComponent(AnchorField, { name: 'a', config: {}, value: 'nowhere', onChange: vi.fn() });
		expect(target.querySelector('[aria-pressed="true"]')?.getAttribute('data-anchor')).toBe('center');
	});

	it('reports the clicked cell', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(AnchorField, { name: 'anchor', config: {}, value: 'center', onChange });

		(target.querySelector('[data-anchor="bottom_right"]') as HTMLButtonElement).click();

		expect(onChange).toHaveBeenCalledWith('anchor', 'bottom_right');
	});

	it('moves one cell with the arrow keys and stops at the edge', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(AnchorField, { name: 'anchor', config: {}, value: 'center', onChange });
		const center = target.querySelector('[data-anchor="center"]') as HTMLButtonElement;

		center.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
		expect(onChange).toHaveBeenLastCalledWith('anchor', 'right');

		const topLeft = target.querySelector('[data-anchor="top_left"]') as HTMLButtonElement;
		topLeft.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		expect(onChange).toHaveBeenLastCalledWith('anchor', 'top_left');

		center.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
		expect(onChange).toHaveBeenLastCalledWith('anchor', 'bottom');
	});

	it('does nothing when disabled', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(AnchorField, { name: 'a', config: { disabled: true }, value: 'center', onChange });
		(target.querySelector('[data-anchor="top"]') as HTMLButtonElement).click();
		expect(onChange).not.toHaveBeenCalled();
	});
});

describe('ColorField', () => {
	const hexInput = (target: HTMLElement) => target.querySelector('input[type="text"]') as HTMLInputElement;

	it('shows the swatch and the hex of a colour', () => {
		const { target } = mountComponent(ColorField, { name: 'key_color', config: { title: 'Key color' }, value: '#00B140', onChange: vi.fn() });

		expect(hexInput(target).value).toBe('#00B140');
		expect((target.querySelector('input[type="color"]') as HTMLInputElement).value).toBe('#00b140');
	});

	it('has no transparent choice unless the param allows one', () => {
		const { target } = mountComponent(ColorField, { name: 'c', config: {}, value: '#112233', onChange: vi.fn() });
		expect(target.textContent).not.toContain('Transparent');
	});

	it('offers Transparent and Colour when allowed, and hides the swatch while transparent', () => {
		const { target } = mountComponent(ColorField, {
			name: 'fill',
			config: { title: 'Fill', allow_transparent: true },
			value: 'transparent',
			onChange: vi.fn()
		});

		const modes = Array.from(target.querySelectorAll('button')).map((button) => button.textContent?.trim());
		expect(modes).toEqual(['Transparent', 'Colour']);
		expect(target.querySelector('[aria-pressed="true"]')?.textContent?.trim()).toBe('Transparent');
		expect(target.querySelector('input[type="color"]')).toBeNull();
		expect(target.querySelector('input[type="text"]')).toBeNull();
	});

	it('picks transparent and then comes back to the colour that was there', () => {
		const onChange = vi.fn();
		const view = mountComponent(ColorField, {
			name: 'fill',
			config: { allow_transparent: true },
			value: '#336699',
			onChange
		});

		(Array.from(view.target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Transparent') as HTMLButtonElement).click();
		expect(onChange).toHaveBeenLastCalledWith('fill', 'transparent');

		view.rerender({ value: 'transparent' });
		(Array.from(view.target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Colour') as HTMLButtonElement).click();
		expect(onChange).toHaveBeenLastCalledWith('fill', '#336699');
	});

	it('comes back to the declared default when no colour was ever held', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(ColorField, {
			name: 'fill',
			config: { allow_transparent: true, default: '#ff0000' },
			value: 'transparent',
			onChange
		});

		(Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Colour') as HTMLButtonElement).click();

		expect(onChange).toHaveBeenLastCalledWith('fill', '#FF0000');
	});

	it('commits a typed hex on blur, upper-cased and expanded from three digits', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(ColorField, { name: 'c', config: {}, value: '#000000', onChange });
		const input = hexInput(target);

		type(input, '#0f0');
		blur(input);

		expect(onChange).toHaveBeenCalledWith('c', '#00FF00');
	});

	it('flags an unfinished hex while typing and puts the last good colour back on blur', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(ColorField, { name: 'c', config: {}, value: '#123456', onChange });
		const input = hexInput(target);

		type(input, '#12zz');
		expect(input.getAttribute('aria-invalid')).toBe('true');
		blur(input);

		expect(onChange).not.toHaveBeenCalled();
		expect(input.value).toBe('#123456');
		expect(input.getAttribute('aria-invalid')).toBe('false');
	});

	it('reports a colour picked on the swatch', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(ColorField, { name: 'c', config: {}, value: '#000000', onChange });
		const swatch = target.querySelector('input[type="color"]') as HTMLInputElement;

		swatch.value = '#abcdef';
		swatch.dispatchEvent(new Event('input', { bubbles: true }));

		expect(onChange).toHaveBeenCalledWith('c', '#ABCDEF');
	});
});

describe('NumberBoxField', () => {
	const box = (target: HTMLElement) => target.querySelector('input') as HTMLInputElement;

	it('shows the value with its unit and an Auto placeholder when empty', () => {
		const { target } = mountComponent(NumberBoxField, { name: 'width', config: { title: 'Width', unit: 'px' }, value: 1920, onChange: vi.fn() });
		expect(box(target).value).toBe('1920');
		expect(target.textContent).toContain('px');

		const empty = mountComponent(NumberBoxField, { name: 'width', config: { title: 'Width' }, value: null, onChange: vi.fn() });
		expect(box(empty.target).value).toBe('');
		expect(box(empty.target).placeholder).toBe('Auto');
	});

	it('reports numbers as they are typed and null when the box is cleared', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(NumberBoxField, { name: 'width', config: {}, value: 100, onChange });

		type(box(target), '640');
		expect(onChange).toHaveBeenLastCalledWith('width', 640);
		type(box(target), '');
		expect(onChange).toHaveBeenLastCalledWith('width', null);
	});

	it('rounds to a whole number when the param is an integer', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(NumberBoxField, { name: 'width', config: { integer: true }, value: 1, onChange });
		type(box(target), '12.6');
		expect(onChange).toHaveBeenLastCalledWith('width', 13);
	});

	it('does not report text that is not a number yet, and says so', () => {
		const onChange = vi.fn();
		const { target } = mountComponent(NumberBoxField, { name: 'width', config: {}, value: 5, onChange });

		type(box(target), 'abc');

		expect(onChange).not.toHaveBeenCalled();
		expect(box(target).getAttribute('aria-invalid')).toBe('true');
		expect(target.textContent).toContain('Enter a number');
	});

	it('names the bound a value breaks', () => {
		const { target } = mountComponent(NumberBoxField, { name: 'w', config: { minimum: 1, maximum: 10 }, value: 0, onChange: vi.fn() });
		expect(target.textContent).toContain('At least 1');
		expect(box(target).getAttribute('aria-invalid')).toBe('true');

		const over = mountComponent(NumberBoxField, { name: 'w', config: { minimum: 1, maximum: 10 }, value: 11, onChange: vi.fn() });
		expect(over.target.textContent).toContain('At most 10');
	});

	it('tidies the text on blur', () => {
		const view = mountComponent(NumberBoxField, {
			name: 'w',
			config: {},
			value: 1,
			onChange: (_name: string, next: unknown) => view.rerender({ value: next })
		});
		type(box(view.target), '007');
		blur(box(view.target));
		expect(box(view.target).value).toBe('7');
	});
});

describe('SliderField with a unit', () => {
	it('writes the unit after the value and on both ends', () => {
		const { target } = mountComponent(SliderField, {
			name: 'scale',
			config: { title: 'Scale', minimum: 1, maximum: 400, step: 1, unit: '%' },
			value: 100,
			onChange: vi.fn()
		});

		expect(target.textContent).toContain('100%');
		expect(target.textContent).toContain('1%');
		expect(target.textContent).toContain('400%');
	});

	it('stays plain when the field declares no unit', () => {
		const { target } = mountComponent(SliderField, {
			name: 'steps',
			config: { title: 'Steps', minimum: 1, maximum: 50 },
			value: 20,
			onChange: vi.fn()
		});
		expect(target.textContent).not.toContain('%');
		expect(target.textContent).toContain('20');
	});

	it('keeps ends at 12px or larger', () => {
		const { target } = mountComponent(SliderField, { name: 's', config: { minimum: 0, maximum: 10 }, value: 1, onChange: vi.fn() });
		expect(target.innerHTML).not.toContain('text-2xs');
	});
});
