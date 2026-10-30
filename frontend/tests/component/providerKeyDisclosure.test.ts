import { describe, it, expect, afterEach, vi, beforeEach } from 'vitest';
import { mount, unmount, flushSync } from 'svelte';

let outcome: 'ok' | 'fail' = 'ok';
const calls: unknown[][] = [];
const saveSetupProviderCredential = {
	mockReset() {
		outcome = 'ok';
		calls.length = 0;
	},
	mockResolvedValue(_value?: unknown) {
		outcome = 'ok';
	},
	failWith() {
		outcome = 'fail';
	}
};

vi.mock('$lib/services/api/index', () => ({
	api: {
		saveSetupProviderCredential: async (...args: unknown[]) => {
			calls.push(args);
			if (outcome === 'fail') throw Object.assign(new Error('nope'), { response: { data: { detail: 'nope' } } });
		}
	}
}));

const { default: ProviderKeyDisclosure } = await import(
	'../../src/lib/components/recipes/ProviderKeyDisclosure.svelte'
);

const PROVIDER = {
	id: 'huggingface',
	name: 'Hugging Face',
	website: 'https://huggingface.co/settings/tokens',
	field_name: 'api_key',
	configured: false
};

let mounted: Record<string, any> | null = null;

function render(provider = PROVIDER, generic = false) {
	const target = document.createElement('div');
	document.body.appendChild(target);
	mounted = mount(ProviderKeyDisclosure, { target, props: { provider, generic } });
	flushSync();
	return target;
}

async function type(target: HTMLElement, value: string) {
	const field = target.querySelector('input[type="password"]') as HTMLInputElement;
	field.value = value;
	field.dispatchEvent(new Event('input', { bubbles: true }));
	flushSync();
}

function saveButton(target: HTMLElement) {
	return Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Save') as HTMLButtonElement;
}

beforeEach(() => saveSetupProviderCredential.mockReset());

afterEach(() => {
	if (mounted) unmount(mounted);
	mounted = null;
	document.body.innerHTML = '';
});

describe('ProviderKeyDisclosure', () => {
	it('uses a generic label and a neutral placeholder in the guided context', () => {
		const target = render(PROVIDER, true);
		expect(target.querySelector('summary')?.textContent?.trim()).toBe('Add a token (optional)');
		expect((target.querySelector('input') as HTMLInputElement).placeholder).toBe('Paste your token');
	});

	it('is collapsed by default and explains what the token is for', () => {
		const target = render();
		const details = target.querySelector('details') as HTMLDetailsElement;
		expect(details.open).toBe(false);
		expect(target.querySelector('summary')?.textContent?.trim()).toBe('Add a Hugging Face token (optional)');
		const text = target.textContent?.replace(/\s+/g, ' ') ?? '';
		expect(text).toContain('Needed for gated models. Can help if downloads fail with rate-limit errors. Stored on this computer.');
		expect(text).not.toMatch(/faster/i);
	});

	it('keeps Save disabled until a token is typed', async () => {
		const target = render();
		expect(saveButton(target).disabled).toBe(true);
		await type(target, 'hf_secret');
		expect(saveButton(target).disabled).toBe(false);
	});

	it('saves the token through the provider credential call and shows the saved state', async () => {
		saveSetupProviderCredential.mockResolvedValue(undefined);
		const target = render();
		await type(target, '  hf_secret ');
		saveButton(target).click();
		await vi.waitFor(() => expect(target.querySelector('[data-provider-key="saved"]')).not.toBeNull());
		expect(calls).toEqual([['huggingface', 'api_key', 'hf_secret']]);
		expect(target.textContent).toContain('Hugging Face API key saved.');
		expect(target.querySelector('input')).toBeNull();
	});

	it('shows an error and keeps the form when saving fails', async () => {
		saveSetupProviderCredential.failWith();
		const target = render();
		await type(target, 'hf_secret');
		saveButton(target).click();
		await vi.waitFor(() => expect(target.querySelector('[role="alert"]')).not.toBeNull());
		expect(target.querySelector('[data-provider-key="saved"]')).toBeNull();
		expect(saveButton(target)).toBeDefined();
	});

	it('shows the saved state straight away for an already configured provider', () => {
		const target = render({ ...PROVIDER, configured: true });
		expect(target.querySelector('details')).toBeNull();
		expect(target.textContent).toContain('Hugging Face API key saved.');
	});
});
