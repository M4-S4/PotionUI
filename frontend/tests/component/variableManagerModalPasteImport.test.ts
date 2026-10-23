// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi } from 'vitest';
import { tick } from 'svelte';
import type { VariablesMap } from '$lib/utils/variableDefs';

const { default: VariableManagerModal } = await import('../../src/lib/components/VariableManagerModal.svelte');
const { createClassComponent } = await import('svelte/legacy');

function mountModal(variables: VariablesMap) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({
		component: VariableManagerModal as never,
		target,
		props: { isOpen: true, variables }
	});
	return { target, component, destroy: () => component.$destroy() };
}

let mounted: ReturnType<typeof mountModal> | undefined;

afterEach(() => {
	mounted?.destroy();
	mounted = undefined;
	document.body.innerHTML = '';
	vi.restoreAllMocks();
	delete (navigator as { clipboard?: unknown }).clipboard;
});

function buttonWithText(text: string): HTMLButtonElement {
	const button = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((b) =>
		b.textContent?.trim().startsWith(text)
	);
	if (!button) throw new Error(`no button with text "${text}"`);
	return button;
}

function textarea(): HTMLTextAreaElement {
	const el = document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Variables JSON"]');
	if (!el) throw new Error('paste textarea not found');
	return el;
}

async function setTextareaValue(el: HTMLTextAreaElement, value: string) {
	el.value = value;
	el.dispatchEvent(new Event('input', { bubbles: true }));
	await tick();
}

describe('VariableManagerModal paste import', () => {
	it('imports pasted JSON and commits the merged map through the change event', async () => {
		Object.defineProperty(navigator, 'clipboard', {
			value: { readText: vi.fn().mockRejectedValue(new Error('denied')) },
			configurable: true
		});

		const variables: VariablesMap = {
			mood: { type: 'text', value: 'existing' }
		};
		mounted = mountModal(variables);

		let latest: VariablesMap | null = null;
		mounted.component.$on('change', (e: CustomEvent<VariablesMap>) => {
			latest = e.detail;
		});

		buttonWithText('Paste').click();
		await tick();

		const pasted = JSON.stringify({
			flavor: { type: 'text', value: 'sweet' },
			mood: { type: 'text', value: 'imported' }
		});
		await setTextareaValue(textarea(), pasted);

		buttonWithText('Import').click();
		await tick();

		expect(latest).not.toBeNull();
		const result = latest as unknown as VariablesMap;
		expect(result.flavor).toEqual({ type: 'text', value: 'sweet' });
		expect(result.mood).toEqual({ type: 'text', value: 'imported' });
	});

	async function openPaste(variables: VariablesMap) {
		Object.defineProperty(navigator, 'clipboard', {
			value: { readText: vi.fn().mockRejectedValue(new Error('denied')) },
			configurable: true
		});
		mounted = mountModal(variables);
		const changes: VariablesMap[] = [];
		mounted.component.$on('change', (e: CustomEvent<VariablesMap>) => changes.push(e.detail));
		buttonWithText('Paste').click();
		await tick();
		return changes;
	}

	it('lists pasted names that collide case-insensitively with each other instead of overwriting', async () => {
		const changes = await openPaste({});
		await setTextareaValue(
			textarea(),
			JSON.stringify({ mood: { type: 'text', value: 'a' }, Mood: { type: 'text', value: 'b' } })
		);
		expect(document.querySelector('[data-testid="paste-notes"]')!.textContent).toContain("'Mood' was skipped");
		buttonWithText('Import').click();
		await tick();
		expect(Object.keys(changes.at(-1)!)).toEqual(['mood']);
		expect(changes.at(-1)!.mood).toEqual({ type: 'text', value: 'a' });
	});

	it('treats a case-variant of an existing name as a clash and replaces the existing entry in place', async () => {
		const changes = await openPaste({ mood: { type: 'text', value: 'old' } });
		await setTextareaValue(textarea(), JSON.stringify({ Mood: { type: 'text', value: 'new' } }));
		expect(document.body.textContent).toContain('already exist');
		buttonWithText('Import').click();
		await tick();
		expect(Object.keys(changes.at(-1)!)).toEqual(['mood']);
		expect(changes.at(-1)!.mood).toEqual({ type: 'text', value: 'new' });
	});
});
