import { describe, expect, it } from 'vitest';
import { getFilledIconPaths, getIconPath, hasIcon } from './IconLibrary';
import { spriteFilledGlyphs, spriteGlyphs } from './iconGlyphs';
import { segmentMenuActions } from './segmentActions';
import { actionCardIcon } from '$lib/chat/actionCard';
import { startersForMode } from '$lib/chat/modeStarters';

const sources = import.meta.glob('/src/**/*.{svelte,ts}', { query: '?raw', import: 'default', eager: true }) as Record<
	string,
	string
>;
const scanned = Object.entries(sources).filter(([path]) => !path.endsWith('iconGlyphs.test.ts'));

function art(name: string) {
	return JSON.stringify([getIconPath(name), getFilledIconPaths(name)]);
}

describe('sprite glyph keys', () => {
	it('keeps the ids that drew different glyphs in different sprites as separate keys', () => {
		for (const id of ['image', 'library', 'pin', 'search', 'trash', 'save']) {
			const keys = ['chat', 'composer', 'panel'].map((p) => `${p}-${id}`).filter(hasIcon);
			const distinct = new Set(keys.map(art));
			expect(distinct.size, id).toBeGreaterThan(1);
		}
	});

	it('shares one definition between sprites that drew the same glyph', () => {
		expect(art('chat-check')).toBe(art('composer-check'));
		expect(art('chat-plus')).toBe(art('panel-plus'));
		expect(art('chat-close')).toBe(art('panel-x'));
	});

	it('draws dots and the play triangle as filled shapes', () => {
		expect(getFilledIconPaths('chat-more')).toHaveLength(3);
		expect(getFilledIconPaths('composer-grip')).toHaveLength(6);
		expect(getFilledIconPaths('composer-play')).toHaveLength(1);
		expect(getFilledIconPaths('composer-trash')).toEqual([]);
	});

	it('registers every glyph in the shared icon library', () => {
		for (const name of Object.keys(spriteGlyphs)) expect(hasIcon(name), name).toBe(true);
		for (const name of Object.keys(spriteFilledGlyphs)) expect(hasIcon(name), name).toBe(true);
	});
});

describe('icon references in the app', () => {
	it('no longer reference an svg sprite', () => {
		const offenders = scanned.filter(([, text]) => /<use\s+href|IconSprite/.test(text)).map(([path]) => path);
		expect(offenders).toEqual([]);
	});

	it('only name sprite glyphs that exist', () => {
		const missing: string[] = [];
		for (const [path, text] of scanned) {
			for (const tag of text.matchAll(/<Icon\s+name=("[^"]*"|\{(?:[^{}]|\$\{[^}]*\})*\})/g)) {
				for (const literal of tag[1].matchAll(/(?:chat|composer|panel)-[a-z][\w-]*/g)) {
					if (!hasIcon(literal[0])) missing.push(`${path}: ${literal[0]}`);
				}
			}
		}
		expect(missing).toEqual([]);
	});

	it('resolves the icon ids produced by data modules', () => {
		const ids = new Set<string>();
		for (const mode of ['generate', 'inspire', 'image', 'library', 'chat', 'director', 'edit']) {
			for (const starter of startersForMode(mode as never)) ids.add(starter.icon);
		}
		const kinds = ['generation', 'text_edit', 'timeline'];
		for (const kind of kinds) ids.add(actionCardIcon({ preview: { action: 'a', items: [], kind } } as never));
		ids.add(actionCardIcon({} as never));
		for (const id of ids) expect(hasIcon(id), id).toBe(true);
	});

	it('resolves the composer glyphs used by segment menu actions', () => {
		const segment = { id: 's', type: 'text', text: '' } as never;
		const actions = segmentMenuActions(segment, { index: 1, total: 3, segmentDisabled: false, hasPromptSyntax: true, hasPromptResources: true });
		const icons = actions.map((a) => a.icon).filter(Boolean) as string[];
		expect(icons.length).toBeGreaterThan(0);
		for (const icon of icons) expect(hasIcon(`composer-${icon}`), icon).toBe(true);
	});
});
