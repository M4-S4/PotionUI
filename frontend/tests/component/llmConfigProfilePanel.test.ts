import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LLMProfile, LLMProfileSchema } from '$lib/types/llm';
import { settle } from './helpers/settle';

const mockGetLLMProfile = vi.fn();
const mockGetLLMProfileSchema = vi.fn();
const mockUpdateLLMProfile = vi.fn();
const mockRefreshLLMProfile = vi.fn();

vi.mock('$lib/services/admin-api', () => ({
	getLLMProfile: (...args: unknown[]) => mockGetLLMProfile(...args),
	getLLMProfileSchema: (...args: unknown[]) => mockGetLLMProfileSchema(...args),
	updateLLMProfile: (...args: unknown[]) => mockUpdateLLMProfile(...args),
	refreshLLMProfile: (...args: unknown[]) => mockRefreshLLMProfile(...args)
}));

const { default: LLMConfigProfilePanel } = await import(
	'../../src/routes/admin/components/llm/LLMConfigProfilePanel.svelte'
);
const { createClassComponent } = await import('svelte/legacy');

const SCHEMA: LLMProfileSchema = {
	fields: [
		{
			name: 'replay',
			kind: 'enum',
			options: ['send_all', 'drop_earlier', 'preserve', 'never'],
			label: 'Reasoning replay',
			group: 'reasoning'
		},
		{ name: 'tool_choice', kind: 'set', options: ['auto', 'none', 'required', 'named'], group: 'tools' },
		{ name: 'context_window', kind: 'int', min: 1024, group: 'limits' },
		{ name: 'vision', kind: 'bool', group: 'tools' },
		{ name: 'reasoning_field', kind: 'string', group: 'reasoning' }
	]
};

function profile(overrides: Partial<LLMProfile> = {}): LLMProfile {
	return {
		config_id: 'c1',
		adapter: 'ollama-chat',
		model: 'qwen3.8:14b',
		discovered_at: '2026-10-27T09:12:00Z',
		discovery_error: null,
		fields: {
			context_window: { value: 65536, source: 'discovered', seed: 32768, discovered: 65536, override: null },
			replay: { value: 'preserve', source: 'override', seed: 'send_all', discovered: null, override: 'preserve' },
			tool_choice: { value: ['auto', 'none'], source: 'seed', seed: ['auto', 'none'], discovered: null, override: null },
			vision: { value: false, source: 'seed', seed: false, discovered: null, override: null },
			reasoning_field: { value: 'reasoning_content', source: 'seed', seed: 'reasoning_content', discovered: null, override: null }
		},
		verified: {
			suite: 'chat_eval:v3',
			passed_at: '2026-10-27',
			runtime: 'ollama 0.13.4',
			gpu_tier: '16gb'
		},
		seed_rows: [
			{ id: 'ollama-qwen3', root: 'marketplace', file: 'ollama.yml' },
			{ id: 'deepseek', root: 'plugin:deepseek-provider', file: 'deepseek.yml' },
			{ id: 'mine', root: 'local', file: 'mine.yml' }
		],
		...overrides
	};
}

let destroy: (() => void) | undefined;

async function mount() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: LLMConfigProfilePanel as never, target, props: { configId: 'c1' } });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	return target;
}

const row = (target: HTMLElement, field: string) =>
	target.querySelector(`[data-testid="profile-row"][data-field="${field}"]`) as HTMLElement;

const footerButton = (target: HTMLElement, text: string) =>
	[...target.querySelectorAll<HTMLButtonElement>('[data-detail-footer] button')].find(
		(button) => button.textContent?.trim() === text
	)!;

async function choose(select: HTMLSelectElement, value: string) {
	select.value = value;
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await settle();
}

async function typeNumber(input: HTMLInputElement, value: string) {
	input.value = value;
	input.dispatchEvent(new Event('input', { bubbles: true }));
	await settle();
}

beforeEach(() => {
	mockGetLLMProfile.mockResolvedValue({ success: true, data: profile() });
	mockGetLLMProfileSchema.mockResolvedValue({ success: true, data: SCHEMA });
	mockUpdateLLMProfile.mockResolvedValue({ success: true, data: profile() });
	mockRefreshLLMProfile.mockResolvedValue({ success: true, data: profile() });
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('LLMConfigProfilePanel', () => {
	it('lists the seed rows with the root and file each came from', async () => {
		const target = await mount();

		const lines = [...target.querySelectorAll('[data-testid="profile-seed-row"]')].map((el) => el.textContent?.trim());
		expect(lines).toEqual([
			'ollama-qwen3 · marketplace/ollama.yml',
			'deepseek · plugin:deepseek-provider/deepseek.yml',
			'mine · local/mine.yml'
		]);
	});

	it('omits the seed rows section when the server sent none', async () => {
		mockGetLLMProfile.mockResolvedValue({ success: true, data: profile({ seed_rows: [] }) });
		const target = await mount();

		expect(target.querySelector('[data-testid="profile-seed-rows"]')).toBeNull();
	});

	it('loads the profile and the schema for the configuration', async () => {
		await mount();

		expect(mockGetLLMProfile).toHaveBeenCalledWith('c1');
		expect(mockGetLLMProfileSchema).toHaveBeenCalledTimes(1);
	});

	it('groups the fields under their schema group', async () => {
		const target = await mount();

		const groups = [...target.querySelectorAll<HTMLElement>('[data-testid="profile-group"]')];
		expect(groups.map((group) => group.dataset.group)).toEqual(['reasoning', 'tools', 'limits']);
		expect([...groups[1].querySelectorAll('[data-testid="profile-row"]')].map((el) => (el as HTMLElement).dataset.field)).toEqual([
			'tool_choice',
			'vision'
		]);
	});

	it('shows each field with its value, its source badge and the layers beneath', async () => {
		const target = await mount();

		expect(row(target, 'context_window').dataset.source).toBe('discovered');
		expect(row(target, 'context_window').textContent).toContain('Discovered');
		expect((row(target, 'context_window').querySelector('input') as HTMLInputElement).value).toBe('65536');
		expect(row(target, 'context_window').querySelector('[data-testid="profile-layers"]')?.textContent).toContain('Seed 32,768');
		expect(row(target, 'replay').dataset.source).toBe('override');
		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('preserve');
		expect(row(target, 'tool_choice').dataset.source).toBe('seed');
	});

	it('renders one control per kind', async () => {
		const target = await mount();

		expect(row(target, 'replay').querySelector('select')).not.toBeNull();
		expect(row(target, 'context_window').querySelector('input[type="number"]')).not.toBeNull();
		expect(row(target, 'reasoning_field').querySelector('input[type="text"]')).not.toBeNull();
		expect(row(target, 'vision').querySelector('input[role="switch"]')).not.toBeNull();
		const chips = [...row(target, 'tool_choice').querySelectorAll<HTMLButtonElement>('button[data-option]')];
		expect(chips.map((chip) => [chip.dataset.option, chip.getAttribute('aria-pressed')])).toEqual([
			['auto', 'true'],
			['none', 'true'],
			['required', 'false'],
			['named', 'false']
		]);
	});

	it('offers a clear button only on an overridden field', async () => {
		const target = await mount();

		expect(row(target, 'replay').querySelector('button[aria-label^="Clear override"]')).not.toBeNull();
		expect(row(target, 'context_window').querySelector('button[aria-label^="Clear override"]')).toBeNull();
	});

	it('starts clean: nothing to save or discard', async () => {
		const target = await mount();

		expect(target.querySelector('[data-detail-footer]')?.textContent).not.toContain('unsaved');
		expect(footerButton(target, 'Save').disabled).toBe(true);
		expect(footerButton(target, 'Discard').disabled).toBe(true);
	});

	it('marks an edited field unsaved and counts it in the footer', async () => {
		const target = await mount();

		await choose(row(target, 'replay').querySelector('select')!, 'never');

		expect(row(target, 'replay').querySelector('[data-testid="profile-unsaved"]')).not.toBeNull();
		expect(row(target, 'replay').dataset.source).toBe('override');
		expect(target.querySelector('[data-detail-footer]')?.textContent).toContain('1 unsaved change');
		expect(footerButton(target, 'Save').disabled).toBe(false);
	});

	it('saves only the changed fields and adopts the returned profile', async () => {
		const saved = profile({
			fields: {
				...profile().fields,
				replay: { value: 'never', source: 'override', seed: 'send_all', discovered: null, override: 'never' }
			}
		});
		mockUpdateLLMProfile.mockResolvedValue({ success: true, data: saved });
		const target = await mount();

		await choose(row(target, 'replay').querySelector('select')!, 'never');
		footerButton(target, 'Save').click();
		await settle();

		expect(mockUpdateLLMProfile).toHaveBeenCalledWith('c1', { replay: 'never' });
		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('never');
		expect(row(target, 'replay').querySelector('[data-testid="profile-unsaved"]')).toBeNull();
		expect(footerButton(target, 'Save').disabled).toBe(true);
	});

	it('sends null to clear a saved override', async () => {
		const target = await mount();

		(row(target, 'replay').querySelector('button[aria-label^="Clear override"]') as HTMLButtonElement).click();
		await settle();

		expect(row(target, 'replay').dataset.source).toBe('seed');
		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('send_all');
		footerButton(target, 'Save').click();
		await settle();

		expect(mockUpdateLLMProfile).toHaveBeenCalledWith('c1', { replay: null });
	});

	it('toggles an option of a set field', async () => {
		const target = await mount();

		(row(target, 'tool_choice').querySelector('button[data-option="required"]') as HTMLButtonElement).click();
		await settle();
		footerButton(target, 'Save').click();
		await settle();

		expect(mockUpdateLLMProfile).toHaveBeenCalledWith('c1', { tool_choice: ['auto', 'none', 'required'] });
	});

	it('edits a number and a switch', async () => {
		const target = await mount();

		await typeNumber(row(target, 'context_window').querySelector('input')!, '131072');
		(row(target, 'vision').querySelector('input[role="switch"]') as HTMLInputElement).click();
		await settle();
		footerButton(target, 'Save').click();
		await settle();

		expect(mockUpdateLLMProfile).toHaveBeenCalledWith('c1', { context_window: 131072, vision: true });
	});

	it('refuses a number below the schema minimum and says why', async () => {
		const target = await mount();

		await typeNumber(row(target, 'context_window').querySelector('input')!, '512');

		expect(row(target, 'context_window').querySelector('[data-testid="profile-problem"]')?.textContent).toBe('Minimum is 1,024');
		expect(footerButton(target, 'Save').disabled).toBe(true);
	});

	it('treats an emptied number box as no edit and restores the value when it loses focus', async () => {
		const target = await mount();
		const input = row(target, 'context_window').querySelector('input')!;

		await typeNumber(input, '');

		expect(row(target, 'context_window').querySelector('[data-testid="profile-unsaved"]')).toBeNull();
		expect(row(target, 'context_window').querySelector('[data-testid="profile-problem"]')).toBeNull();
		expect(footerButton(target, 'Save').disabled).toBe(true);
		input.dispatchEvent(new Event('blur'));
		await settle();
		expect(input.value).toBe('65536');
	});

	it('discards every edit', async () => {
		const target = await mount();

		await choose(row(target, 'replay').querySelector('select')!, 'never');
		footerButton(target, 'Discard').click();
		await settle();

		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('preserve');
		expect(mockUpdateLLMProfile).not.toHaveBeenCalled();
		expect(footerButton(target, 'Save').disabled).toBe(true);
	});

	it('keeps the edits and the form when saving fails', async () => {
		mockUpdateLLMProfile.mockResolvedValue({ success: false, message: 'replay is not valid for this adapter' });
		const target = await mount();

		await choose(row(target, 'replay').querySelector('select')!, 'never');
		footerButton(target, 'Save').click();
		await settle();

		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('never');
		expect(target.querySelector('[data-detail-footer]')?.textContent).toContain('1 unsaved change');
	});

	it('refreshes discovery and shows the new values without losing an edit', async () => {
		mockRefreshLLMProfile.mockResolvedValue({
			success: true,
			data: profile({
				discovered_at: '2026-10-28T10:00:00Z',
				fields: {
					...profile().fields,
					context_window: { value: 131072, source: 'discovered', seed: 32768, discovered: 131072, override: null }
				}
			})
		});
		const target = await mount();
		await choose(row(target, 'replay').querySelector('select')!, 'never');

		const refresh = [...target.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Refresh')!;
		refresh.click();
		await settle();

		expect(mockRefreshLLMProfile).toHaveBeenCalledWith('c1');
		expect((row(target, 'context_window').querySelector('input') as HTMLInputElement).value).toBe('131072');
		expect((row(target, 'replay').querySelector('select') as HTMLSelectElement).value).toBe('never');
	});

	it('shows the verified badge with suite, date, runtime and GPU tier', async () => {
		const target = await mount();

		const verified = target.querySelector('[data-testid="profile-verified"]');
		expect(verified?.textContent).toContain('Verified');
		const text = target.textContent ?? '';
		expect(text).toContain('chat_eval:v3');
		expect(text).toContain('2026-10-27');
		expect(text).toContain('ollama 0.13.4');
		expect(text).toContain('16gb');
	});

	it('says the profile is not verified when no suite has passed', async () => {
		mockGetLLMProfile.mockResolvedValue({ success: true, data: profile({ verified: null }) });
		const target = await mount();

		expect(target.querySelector('[data-testid="profile-verified"]')).toBeNull();
		expect(target.querySelector('[data-testid="profile-unverified"]')?.textContent).toContain('Not verified');
	});

	it('shows why discovery failed and keeps the seed values readable', async () => {
		mockGetLLMProfile.mockResolvedValue({
			success: true,
			data: profile({ discovery_error: 'connection refused', discovered_at: null })
		});
		const target = await mount();

		expect(target.querySelector('[data-testid="profile-discovery-error"]')?.textContent).toBe('connection refused');
		expect(target.textContent).toContain('Never');
		expect(row(target, 'tool_choice')).not.toBeNull();
	});

	it('shows a field the schema does not know as read-only text', async () => {
		mockGetLLMProfile.mockResolvedValue({
			success: true,
			data: profile({
				fields: {
					...profile().fields,
					prices: { value: { input: 3 }, source: 'seed', seed: { input: 3 }, discovered: null, override: null }
				}
			})
		});
		const target = await mount();

		const unknown = row(target, 'prices');
		expect(unknown.querySelector('input, select')).toBeNull();
		expect(unknown.querySelector('[data-testid="profile-readonly"]')?.textContent).toBe('{"input":3}');
	});

	it('offers a retry when the profile cannot be loaded', async () => {
		mockGetLLMProfile.mockRejectedValueOnce(new Error('network down'));
		const target = await mount();

		expect(target.textContent).toContain("Couldn't load");
		const retry = [...target.querySelectorAll('button')].find((button) => button.textContent?.trim() === 'Retry')!;
		retry.click();
		await settle();

		expect(mockGetLLMProfile).toHaveBeenCalledTimes(2);
		expect(row(target, 'replay')).not.toBeNull();
	});

	it('uses no native title tooltip', async () => {
		const target = await mount();

		expect(target.querySelector('[title]')).toBeNull();
	});
});
