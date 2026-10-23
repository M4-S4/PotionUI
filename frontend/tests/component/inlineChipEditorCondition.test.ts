// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { flushSync, tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: InlineChipEditor } = await import('$lib/components/InlineChipEditor.svelte');
const { createClassComponent } = await import('svelte/legacy');

let target: HTMLDivElement;
let component: ReturnType<typeof createClassComponent> | undefined;

const variables: VariablesMap = {
	age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 18 }
};

function mountEditor(value: string, vars: VariablesMap = variables) {
	target = document.createElement('div');
	document.body.appendChild(target);
	component = createClassComponent({
		component: InlineChipEditor as never,
		target,
		props: { value, chips: {}, variant: 'segment-composer', borderless: true, variables: vars }
	});
	const values: string[] = [];
	const handle = component as unknown as {
		$on: (n: string, cb: (e: CustomEvent) => void) => void;
		$set: (props: Record<string, unknown>) => void;
	};
	handle.$on('change', (e) => {
		values.push(e.detail.value);
		handle.$set({ value: e.detail.value });
	});
	return {
		editor: component as unknown as { insertCondition: () => void },
		values
	};
}

afterEach(() => {
	component?.$destroy();
	component = undefined;
	target?.remove();
	document.body.innerHTML = '';
});

describe('InlineChipEditor conditions', () => {
	it('renders a block in the text as one condition chip and no raw grammar', () => {
		mountEditor('woman, <<if $age > 65>>grey hair<<else>>dark<<end>>, soft light');
		flushSync();
		expect(target.querySelectorAll('.condition-chip').length).toBe(1);
		expect(target.querySelector('.choice-group-chip')).toBeNull();
		const container = target.querySelector<HTMLElement>('.choice-group-container')!;
		expect(container.dataset.groupRaw).toBe('<<if $age > 65>>grey hair<<else>>dark<<end>>');
		expect(target.querySelector('.inline-chip-editor')!.textContent).not.toContain('<<');
	});

	it('leaves a block that never closes as plain text', () => {
		mountEditor('woman, <<if $age > 65>>grey hair');
		flushSync();
		expect(target.querySelector('.condition-chip')).toBeNull();
		expect(target.querySelector('.inline-chip-editor')!.textContent).toContain('<<if $age > 65>>');
	});

	it('insertCondition writes a block for the first variable, chip-ifies it and opens its modal', async () => {
		const { editor, values } = mountEditor('');
		editor.insertCondition();
		flushSync();
		await tick();
		expect(values.at(-1)).toBe('<<if $age >= 18>>text<<end>>');
		expect(target.querySelectorAll('.condition-chip').length).toBe(1);
		expect(document.querySelector('[role="dialog"][aria-labelledby="condition-modal-title"]')).not.toBeNull();
	});

	it('saving in the modal updates the text without remounting the chip', async () => {
		const { editor, values } = mountEditor('');
		editor.insertCondition();
		flushSync();
		await tick();
		const chip = target.querySelector('.condition-chip')!;
		const pop = document.querySelector('[role="dialog"][aria-labelledby="condition-modal-title"]')!;
		const area = pop.querySelector<HTMLTextAreaElement>('textarea[aria-label="If text"]')!;
		area.value = 'grey hair';
		area.dispatchEvent(new Event('input', { bubbles: true }));
		await tick();
		Array.from(pop.querySelectorAll('button')).find((b) => b.textContent?.trim().startsWith('Save'))!.click();
		await tick();
		expect(values.at(-1)).toBe('<<if $age >= 18>>grey hair<<end>>');
		expect(target.querySelector('.condition-chip')).toBe(chip);
	});

	it('removing the chip removes the block from the text', async () => {
		const { values } = mountEditor('a <<if $age > 65>>x<<end>> b');
		flushSync();
		await tick();
		target.querySelector('.chip-main')!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
		await tick();
		const pop = document.querySelector('[role="dialog"][aria-labelledby="condition-modal-title"]')!;
		Array.from(pop.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Remove condition')!.click();
		await tick();
		expect(values.at(-1)).toBe('a  b');
		expect(target.querySelector('.condition-chip')).toBeNull();
	});
});
