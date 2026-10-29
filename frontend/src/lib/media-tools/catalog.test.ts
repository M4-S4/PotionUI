import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get, writable } from 'svelte/store';

const listMediaTools = vi.fn();
vi.mock('$lib/services/api/mediaTools', () => ({
	listMediaTools: (...args: unknown[]) => listMediaTools(...args)
}));

const extensionRegistrations = writable(0);
vi.mock('$lib/plugin-api/extensionRefresh', () => ({ extensionRegistrations }));

import {
	currentCatalog,
	isRunnable,
	loadMediaToolCatalog,
	mediaToolCatalog,
	mediaToolCatalogRevision,
	refreshMediaToolCatalog,
	rememberToolParams,
	resetMediaToolCatalog,
	toolAvailability,
	toolById,
	toolsFor
} from './catalog';
import type { MediaToolSource, ToolDTO } from './types';

function tool(overrides: Partial<ToolDTO> & { id: string }): ToolDTO {
	return {
		label: overrides.id,
		icon: 'sliders',
		accepts: ['image'],
		produces: 'same',
		cardinality: 'one_to_one',
		params: [],
		requires: {},
		selection: null,
		scopes: ['library', 'history', 'field'],
		editor: null,
		preview: false,
		component: null,
		source: 'core',
		last_params: null,
		...overrides
	};
}

const trim = tool({ id: 'trim', accepts: ['video', 'audio'], editor: 'trim' });
const rotate = tool({ id: 'rotate', accepts: ['image', 'video'], preview: true });
const resize = tool({ id: 'resize', accepts: ['image', 'video'], scopes: ['library', 'history'] });
const concat = tool({ id: 'concat', accepts: ['video'], cardinality: 'many_to_one', selection: { min: 2, max: 8 } });
const cropSubject = tool({ id: 'crop_subject', accepts: ['image'], requires: { has_alpha: true } });
const ALL = [trim, rotate, resize, concat, cropSubject];

function source(overrides: Partial<MediaToolSource> = {}): MediaToolSource {
	return { id: 's1', kind: 'image', url: '/a.png', filename: 'a.png', ...overrides };
}

beforeEach(() => {
	listMediaTools.mockReset();
	extensionRegistrations.set(0);
	resetMediaToolCatalog();
});

describe('loading', () => {
	it('fetches the whole catalog once and serves it from memory after that', async () => {
		listMediaTools.mockResolvedValue(ALL);

		const first = await loadMediaToolCatalog();
		const second = await loadMediaToolCatalog();

		expect(listMediaTools).toHaveBeenCalledTimes(1);
		expect(listMediaTools).toHaveBeenCalledWith();
		expect(first).toHaveLength(5);
		expect(second).toBe(first);
		expect(get(mediaToolCatalog)).toHaveLength(5);
	});

	it('shares one request between callers that arrive together', async () => {
		listMediaTools.mockResolvedValue(ALL);
		await Promise.all([loadMediaToolCatalog(), loadMediaToolCatalog(), loadMediaToolCatalog()]);
		expect(listMediaTools).toHaveBeenCalledTimes(1);
	});

	it('answers with an empty list when the request fails, and tries again next time', async () => {
		listMediaTools.mockRejectedValueOnce(new Error('offline'));
		expect(await loadMediaToolCatalog()).toEqual([]);

		listMediaTools.mockResolvedValue(ALL);
		expect(await loadMediaToolCatalog()).toHaveLength(5);
		expect(listMediaTools).toHaveBeenCalledTimes(2);
	});

	it('ticks the revision each time a catalog is applied', async () => {
		listMediaTools.mockResolvedValue(ALL);
		await loadMediaToolCatalog();
		expect(get(mediaToolCatalogRevision)).toBe(1);
		await refreshMediaToolCatalog();
		expect(get(mediaToolCatalogRevision)).toBe(2);
	});

	it('reloads when the plugin set changes, and not for the initial value', async () => {
		listMediaTools.mockResolvedValue(ALL);
		await loadMediaToolCatalog();
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(listMediaTools).toHaveBeenCalledTimes(1);

		extensionRegistrations.set(1);
		await vi.waitFor(() => expect(listMediaTools).toHaveBeenCalledTimes(2));
	});
});

describe('toolsFor', () => {
	it('keeps the tools that accept the kind and are offered in the scope', () => {
		expect(toolsFor(['image'], 'library', ALL).map((entry) => entry.id)).toEqual(['rotate', 'resize', 'crop_subject']);
		expect(toolsFor(['image'], 'field', ALL).map((entry) => entry.id)).toEqual(['rotate', 'crop_subject']);
	});

	it('needs a tool to accept every selected kind', () => {
		expect(toolsFor(['image', 'video'], 'library', ALL).map((entry) => entry.id)).toEqual(['rotate', 'resize']);
		expect(toolsFor(['audio', 'video'], 'library', ALL).map((entry) => entry.id)).toEqual(['trim']);
	});

	it('offers nothing for no kinds', () => {
		expect(toolsFor([], 'library', ALL)).toEqual([]);
	});

	it('treats a tool with no scopes as offered everywhere', () => {
		const everywhere = tool({ id: 'x', scopes: [] });
		expect(toolsFor(['image'], 'history', [everywhere])).toHaveLength(1);
	});

	it('reads the loaded catalog by default', async () => {
		listMediaTools.mockResolvedValue(ALL);
		await loadMediaToolCatalog();
		expect(toolsFor(new Set(['audio']), 'library').map((entry) => entry.id)).toEqual(['trim']);
		expect(toolById('concat')?.cardinality).toBe('many_to_one');
		expect(toolById('nope')).toBeNull();
	});
});

describe('rememberToolParams', () => {
	it('keeps what a finished run used so the next dialog opens on it', async () => {
		listMediaTools.mockResolvedValue(ALL);
		await loadMediaToolCatalog();

		rememberToolParams('rotate', { degrees: 90 });

		expect(toolById('rotate')?.last_params).toEqual({ degrees: 90 });
		expect(currentCatalog().find((entry) => entry.id === 'trim')?.last_params).toBeNull();
	});
});

describe('isRunnable', () => {
	it('treats a component with no params as a surface of its own', () => {
		expect(isRunnable(tool({ id: 'a', component: 'plugin:p:view' }))).toBe(false);
	});

	it('runs anything with params, an editor or no component', () => {
		expect(isRunnable(tool({ id: 'a', component: 'plugin:p:view', params: [{ name: 'x', type: 'integer' }] }))).toBe(true);
		expect(isRunnable(tool({ id: 'a', component: 'plugin:p:view', editor: 'trim' }))).toBe(true);
		expect(isRunnable(tool({ id: 'a' }))).toBe(true);
	});

	it('trusts an explicit flag over the guess', () => {
		expect(isRunnable(tool({ id: 'a', component: 'plugin:p:view', runnable: true }))).toBe(true);
		expect(isRunnable(tool({ id: 'a', runnable: false }))).toBe(false);
	});
});

describe('toolAvailability', () => {
	it('needs a selection', () => {
		expect(toolAvailability(rotate, [])).toEqual({ enabled: false, reason: 'Select an item' });
	});

	it('enforces the selection bounds of a many-to-one tool', () => {
		const clips = [source({ kind: 'video' }), source({ id: 's2', kind: 'video' })];
		expect(toolAvailability(concat, clips.slice(0, 1))).toEqual({ enabled: false, reason: 'Select 2 to 8 items' });
		expect(toolAvailability(concat, clips)).toEqual({ enabled: true });
		const nine = Array.from({ length: 9 }, (_, index) => source({ id: `c${index}`, kind: 'video' }));
		expect(toolAvailability(concat, nine).enabled).toBe(false);
	});

	it('lets a one-to-one tool take several items', () => {
		expect(toolAvailability(rotate, [source(), source({ id: 's2' })])).toEqual({ enabled: true });
	});

	it('refuses a kind the tool does not accept', () => {
		expect(toolAvailability(rotate, [source({ kind: 'audio' })])).toEqual({
			enabled: false,
			reason: 'Not available for audio'
		});
	});

	it('asks for transparency from a jpeg but gives a png the benefit of the doubt', () => {
		expect(toolAvailability(cropSubject, [source({ filename: 'a.jpg', url: '/a.jpg' })])).toEqual({
			enabled: false,
			reason: 'Needs a transparent image'
		});
		expect(toolAvailability(cropSubject, [source()])).toEqual({ enabled: true });
		expect(toolAvailability(cropSubject, [source({ hasAlpha: false })]).enabled).toBe(false);
		expect(toolAvailability(cropSubject, [source({ filename: 'a.jpg', hasAlpha: true })]).enabled).toBe(true);
	});

	it('checks the minimum duration only when the duration is known', () => {
		const long = tool({ id: 'x', accepts: ['video'], requires: { min_duration: 2 } });
		expect(toolAvailability(long, [source({ kind: 'video', durationSeconds: 1 })])).toEqual({
			enabled: false,
			reason: 'Needs at least 2s of media'
		});
		expect(toolAvailability(long, [source({ kind: 'video' })])).toEqual({ enabled: true });
	});

	it('refuses a silent clip for a tool that needs sound', () => {
		const sound = tool({ id: 'x', accepts: ['video'], requires: { has_audio: true } });
		expect(toolAvailability(sound, [source({ kind: 'video', hasAudio: false })]).enabled).toBe(false);
		expect(toolAvailability(sound, [source({ kind: 'video', hasAudio: true })]).enabled).toBe(true);
	});
});
