import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LLMConfig, LLMProfile, LLMProfileSchema } from '$lib/types/llm';
import { settle } from './helpers/settle';

const mockGetLLMConfigurations = vi.fn();
const mockGetPreChatActions = vi.fn();
const mockGetLLMAssignmentSummary = vi.fn();
const mockGetLLMProfile = vi.fn();
const mockGetLLMProfileSchema = vi.fn();
const mockConfirm = vi.fn();

vi.mock('$lib/services/api/index', () => ({
	api: {
		getLLMConfigurations: (...args: unknown[]) => mockGetLLMConfigurations(...args),
		getPreChatActions: (...args: unknown[]) => mockGetPreChatActions(...args),
		getLLMProviderTypes: vi.fn(async () => ({ success: true, data: { types: [] } })),
		listNativeCheckpoints: vi.fn()
	}
}));

vi.mock('$lib/services/admin-api', () => ({
	getLLMAssignmentSummary: (...args: unknown[]) => mockGetLLMAssignmentSummary(...args),
	getLLMProfile: (...args: unknown[]) => mockGetLLMProfile(...args),
	getLLMProfileSchema: (...args: unknown[]) => mockGetLLMProfileSchema(...args),
	updateLLMProfile: vi.fn(),
	refreshLLMProfile: vi.fn(),
	updateLLMConfiguration: vi.fn(),
	createLLMConfiguration: vi.fn(),
	deleteLLMConfiguration: vi.fn()
}));

vi.mock('$lib/stores/confirm', () => ({
	confirmDialog: (...args: unknown[]) => mockConfirm(...args),
	activeConfirm: { subscribe: () => () => {} },
	resolveActiveConfirm: vi.fn()
}));

const { default: LLMConfigTab } = await import('../../src/routes/admin/components/LLMConfigTab.svelte');
const { createClassComponent } = await import('svelte/legacy');

const SCHEMA: LLMProfileSchema = {
	fields: [
		{ name: 'replay', kind: 'enum', options: ['send_all', 'drop_earlier', 'preserve', 'never'], group: 'reasoning' }
	]
};

const PROFILE: LLMProfile = {
	config_id: 'cfg-1',
	adapter: 'anthropic-messages',
	model: 'claude-sonnet-5-5',
	discovered_at: null,
	discovery_error: null,
	fields: {
		replay: { value: 'send_all', source: 'seed', seed: 'send_all', discovered: null, override: null }
	},
	verified: null
};

function config(overrides: Partial<LLMConfig> = {}): LLMConfig {
	return {
		id: 'cfg-1',
		name: 'Claude',
		type: 'anthropic',
		enabled: true,
		base_url: 'https://api.anthropic.com',
		api_key_set: true,
		model: 'claude-sonnet-5-5',
		system_message: 'Be terse.',
		temperature: 0.7,
		max_tokens: 1000,
		timeout: 30,
		provider_options: {},
		...overrides
	};
}

let destroy: (() => void) | undefined;

async function open() {
	const target = document.createElement('div');
	document.body.appendChild(target);
	const component = createClassComponent({ component: LLMConfigTab as never, target, props: {} });
	destroy = () => {
		component.$destroy();
		target.remove();
	};
	await settle();
	(target.querySelector('.dt-scroll > [role="row"]') as HTMLElement).click();
	await settle();
	return target;
}

const tab = (target: HTMLElement, label: string) =>
	[...target.querySelectorAll<HTMLButtonElement>('nav button')].find((button) => button.textContent?.trim() === label)!;

async function openProfile(target: HTMLElement) {
	tab(target, 'Profile').click();
	await settle();
}

async function editReplay(target: HTMLElement) {
	const select = target.querySelector<HTMLSelectElement>('#profile-replay')!;
	select.value = 'never';
	select.dispatchEvent(new Event('change', { bubbles: true }));
	await settle();
}

const back = (target: HTMLElement) => target.querySelector<HTMLButtonElement>('button[aria-label="Back to Configurations"]')!;

beforeEach(() => {
	mockGetLLMConfigurations.mockResolvedValue({ success: true, data: { configurations: [config()] } });
	mockGetPreChatActions.mockResolvedValue({ success: true, data: { actions: [] } });
	mockGetLLMAssignmentSummary.mockResolvedValue({ success: true, data: {} });
	mockGetLLMProfile.mockResolvedValue({ success: true, data: PROFILE });
	mockGetLLMProfileSchema.mockResolvedValue({ success: true, data: SCHEMA });
	mockConfirm.mockResolvedValue(false);
});

afterEach(() => {
	destroy?.();
	destroy = undefined;
	document.body.innerHTML = '';
	vi.clearAllMocks();
});

describe('LLMConfigTab capability profile', () => {
	it('lists a Profile tab between Configuration and Toolset', async () => {
		const target = await open();

		const labels = [...target.querySelectorAll('nav button')].map((button) => button.textContent?.trim().replace(/\d+$/, '').trim());
		expect(labels).toEqual(['Configuration', 'Profile', 'Toolset', 'Access']);
	});

	it('does not build the profile editor until the tab is opened', async () => {
		const target = await open();

		expect(mockGetLLMProfileSchema).not.toHaveBeenCalled();
		expect(target.querySelector('[data-testid="profile-row"]')).toBeNull();
	});

	it('loads the profile of the selected configuration when the tab opens', async () => {
		const target = await open();

		await openProfile(target);

		expect(mockGetLLMProfile).toHaveBeenCalledWith('cfg-1');
		expect(target.querySelector('#profile-replay')).not.toBeNull();
		expect(target.textContent).toContain('anthropic-messages');
	});

	it('asks before leaving the configuration with unsaved profile edits', async () => {
		const target = await open();
		await openProfile(target);
		await editReplay(target);

		back(target).click();
		await settle();

		expect(mockConfirm).toHaveBeenCalledTimes(1);
		expect(back(target)).not.toBeNull();
	});

	it('leaves without asking when the profile was not touched', async () => {
		const target = await open();
		await openProfile(target);

		back(target).click();
		await settle();

		expect(mockConfirm).not.toHaveBeenCalled();
		expect(target.querySelector('button[aria-label="Back to Configurations"]')).toBeNull();
	});

	it('keeps an unsaved profile edit while another tab is shown', async () => {
		const target = await open();
		await openProfile(target);
		await editReplay(target);

		tab(target, 'Configuration').click();
		await settle();
		await openProfile(target);

		expect(target.querySelector<HTMLSelectElement>('#profile-replay')!.value).toBe('never');
		expect(target.querySelector('[data-detail-footer]')?.textContent).toContain('1 unsaved change');
	});

	it('shows the Configuration footer only on the Configuration tab', async () => {
		const target = await open();
		expect(target.querySelectorAll('[data-detail-footer]')).toHaveLength(1);

		await openProfile(target);

		expect(target.querySelectorAll('[data-detail-footer]')).toHaveLength(1);
		tab(target, 'Toolset').click();
		await settle();
		expect(target.querySelectorAll('[data-detail-footer]')).toHaveLength(0);
	});
});
