import { vi } from 'vitest';
import { getRegistry, registerComponent } from '$lib/plugin-api/componentRegistry';

const HOST_COMPONENTS: Record<string, () => Promise<{ default: any }>> = {
	Alert: () => import('$lib/components/ui/Alert.svelte'),
	Badge: () => import('$lib/components/ui/Badge.svelte'),
	Button: () => import('$lib/components/ui/Button.svelte'),
	IconButton: () => import('$lib/components/ui/IconButton.svelte'),
	SegmentedControl: () => import('$lib/components/ui/SegmentedControl.svelte'),
	Spinner: () => import('$lib/components/ui/Spinner.svelte'),
	Switch: () => import('$lib/components/ui/Switch.svelte'),
	Tooltip: () => import('$lib/components/Tooltip.svelte')
};

export interface PluginHostStubs {
	confirm: ReturnType<typeof vi.fn>;
	toast: ReturnType<typeof vi.fn>;
}

export async function installPluginHost(extra: Record<string, unknown> = {}): Promise<PluginHostStubs> {
	for (const [name, load] of Object.entries(HOST_COMPONENTS)) {
		registerComponent(name, (await load()).default);
	}
	const confirm = vi.fn(async () => true);
	const toast = vi.fn();
	(window as unknown as { __potionui: unknown }).__potionui = {
		components: getRegistry(),
		api: { fetch: (url: string, init?: RequestInit) => globalThis.fetch(url, init) },
		confirm,
		notifications: { toast },
		...extra
	};
	return { confirm, toast };
}

export function removePluginHost(): void {
	delete (window as unknown as { __potionui?: unknown }).__potionui;
}
