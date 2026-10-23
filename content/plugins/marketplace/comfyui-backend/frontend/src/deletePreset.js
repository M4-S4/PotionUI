import { call, messageOf } from './api.js';

export async function deletePreset(apiBase, preset) {
	const host = globalThis.__potionui;
	const confirmed = await host.confirm({
		title: `Delete "${preset.name}"?`,
		message: `This removes ${preset.family}/${preset.variant} from content/presets/local permanently. This can't be undone.`,
		variant: 'danger',
		confirmLabel: 'Delete'
	});
	if (!confirmed) return false;
	try {
		const { ok, status, payload } = await call(`${apiBase}/presets/imported/${preset.preset_id}`, {
			method: 'DELETE'
		});
		if (!ok) {
			host.notifications.toast('error', messageOf(payload, `Delete failed (${status})`));
			return false;
		}
		return true;
	} catch (e) {
		host.notifications.toast('error', 'Could not reach the server.');
		return false;
	}
}
