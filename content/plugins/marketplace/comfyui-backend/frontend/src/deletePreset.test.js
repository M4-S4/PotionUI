import { test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { deletePreset } from './deletePreset.js';

const preset = { preset_id: 'p1', name: 'My flow', family: 'Qwen', variant: 'imported' };

let toasts;
let requests;
let saved;

function install({ confirmed = true, reply }) {
	toasts = [];
	requests = [];
	globalThis.__potionui = {
		confirm: async (options) => {
			requests.push({ confirm: options });
			return confirmed;
		},
		notifications: { toast: (level, message) => toasts.push([level, message]) },
		api: {
			fetch: async (url, init) => {
				requests.push({ url, init });
				if (reply instanceof Error) throw reply;
				return { ok: reply.status < 400, status: reply.status, json: async () => reply.body };
			}
		}
	};
}

beforeEach(() => {
	saved = globalThis.__potionui;
});

afterEach(() => {
	if (saved === undefined) delete globalThis.__potionui;
	else globalThis.__potionui = saved;
});

test('a declined confirmation sends no request and shows no toast', async () => {
	install({ confirmed: false, reply: { status: 200, body: {} } });
	assert.equal(await deletePreset('/api/plugins/comfyui-backend', preset), false);
	assert.equal(requests.filter((r) => r.url).length, 0);
	assert.deepEqual(toasts, []);
});

test('the confirmation is a danger dialog naming the preset', async () => {
	install({ reply: { status: 200, body: {} } });
	await deletePreset('/api/plugins/comfyui-backend', preset);
	const { confirm } = requests[0];
	assert.equal(confirm.variant, 'danger');
	assert.match(confirm.title, /My flow/);
	assert.match(confirm.message, /Qwen\/imported/);
});

test('a confirmed delete sends DELETE and reports success', async () => {
	install({ reply: { status: 200, body: {} } });
	assert.equal(await deletePreset('/api/plugins/comfyui-backend', preset), true);
	const call = requests.find((r) => r.url);
	assert.equal(call.url, '/api/plugins/comfyui-backend/presets/imported/p1');
	assert.equal(call.init.method, 'DELETE');
	assert.deepEqual(toasts, []);
});

test('a server refusal toasts the failure text and keeps the row', async () => {
	install({ reply: { status: 409, body: { detail: 'Preset is in use' } } });
	assert.equal(await deletePreset('/api/plugins/comfyui-backend', preset), false);
	assert.deepEqual(toasts, [['error', 'Preset is in use']]);
});

test('a refusal without text falls back to the status', async () => {
	install({ reply: { status: 500, body: null } });
	await deletePreset('/api/plugins/comfyui-backend', preset);
	assert.deepEqual(toasts, [['error', 'Delete failed (500)']]);
});

test('an unreachable server toasts and keeps the row', async () => {
	install({ reply: new TypeError('Failed to fetch') });
	assert.equal(await deletePreset('/api/plugins/comfyui-backend', preset), false);
	assert.deepEqual(toasts, [['error', 'Could not reach the server.']]);
});
