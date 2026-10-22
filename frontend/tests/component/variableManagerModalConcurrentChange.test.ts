// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableManagerModal } = await import('../../src/lib/components/VariableManagerModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

const style = { type: 'text', value: 'oil' } as const;
const mood = { type: 'text', value: 'noir' } as const;

function mountModal(variables: VariablesMap) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: VariableManagerModal as never,
		target,
		props: { isOpen: true, variables }
	});
	const changes: VariablesMap[] = [];
	component.$on('change', (event: CustomEvent<VariablesMap>) => changes.push(event.detail));
	return { component, changes, destroy: () => component.$destroy() };
}

let mounted: ReturnType<typeof mountModal> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	document.body.innerHTML = '';
});

function buttonWithText(text: string): HTMLButtonElement {
	const button = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
		b.textContent?.trim().startsWith(text)
	);
	if (!button) throw new Error(`no button with text "${text}"`);
	return button;
}

async function typeValue(value: string) {
	const input = document.querySelector<HTMLInputElement>('input[aria-label="Variable value"]');
	if (!input) throw new Error('value input not found');
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

describe('VariableManagerModal with a change arriving while open', () => {
	it('keeps a variable that arrived while open and applies only the edited one on Done', async () => {
		mounted = mountModal({ style });
		await tick();

		mounted.component.$set({ variables: { style, mood } });
		await tick();
		await typeValue('watercolor');
		buttonWithText('Done').click();
		await tick();

		expect(mounted.changes.at(-1)).toEqual({
			style: { type: 'text', value: 'watercolor' },
			mood
		});
	});

	it('does not resurrect a variable removed elsewhere while open when it was untouched', async () => {
		mounted = mountModal({ style, mood });
		await tick();

		mounted.component.$set({ variables: { style } });
		await tick();
		await typeValue('watercolor');

		expect(mounted.changes.at(-1)).toEqual({ style: { type: 'text', value: 'watercolor' } });
	});

	it('cancel reverts only the modal edits and keeps what arrived meanwhile', async () => {
		mounted = mountModal({ style });
		await tick();

		mounted.component.$set({ variables: { style, mood } });
		await tick();
		await typeValue('watercolor');
		buttonWithText('Cancel').click();
		await tick();
		buttonWithText('Discard').click();
		await tick();

		expect(mounted.changes.at(-1)).toEqual({ style, mood });
	});
});
