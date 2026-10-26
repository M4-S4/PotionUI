import { describe, expect, it } from 'vitest';
import type { LLMProfile, LLMProfileSchema } from '$lib/types/llm';
import {
	buildProfileGroups,
	clearProfileDraft,
	formatProfileValue,
	humanizeProfileName,
	profileDraftCount,
	profileDraftPayload,
	profileHasProblems,
	profileLayers,
	sameProfileValue,
	setProfileDraft,
	type ProfileRow
} from './llmProfile';

const SCHEMA: LLMProfileSchema = {
	fields: [
		{ name: 'replay', kind: 'enum', options: ['send_all', 'drop_earlier', 'preserve', 'never'], label: 'Reasoning replay', group: 'reasoning' },
		{ name: 'tool_choice', kind: 'set', options: ['auto', 'none', 'required', 'named'], group: 'tools' },
		{ name: 'context_window', kind: 'int', min: 1024, group: 'limits' },
		{ name: 'vision', kind: 'bool', group: 'tools' }
	]
};

const PROFILE: LLMProfile = {
	config_id: 'c1',
	adapter: 'ollama-chat',
	model: 'qwen3.8:14b',
	discovered_at: '2026-10-27T09:12:00Z',
	discovery_error: null,
	fields: {
		context_window: { value: 65536, source: 'discovered', seed: 32768, discovered: 65536, override: null },
		replay: { value: 'preserve', source: 'override', seed: 'send_all', discovered: null, override: 'preserve' },
		tool_choice: { value: ['auto', 'none'], source: 'seed', seed: ['auto', 'none'], discovered: null, override: null },
		vision: { value: false, source: 'seed', seed: false, discovered: null, override: null }
	},
	verified: null
};

const rowOf = (draft = {}, name = 'context_window'): ProfileRow =>
	buildProfileGroups(PROFILE, SCHEMA, draft)
		.flatMap((group) => group.rows)
		.find((row) => row.name === name)!;

describe('humanizeProfileName', () => {
	it('turns a field name into a label', () => {
		expect(humanizeProfileName('context_window')).toBe('Context window');
		expect(humanizeProfileName('tool-choice')).toBe('Tool choice');
	});
});

describe('buildProfileGroups', () => {
	it('groups the fields by schema group in schema order', () => {
		const groups = buildProfileGroups(PROFILE, SCHEMA, {});

		expect(groups.map((group) => [group.key, group.label, group.rows.map((row) => row.name)])).toEqual([
			['reasoning', 'Reasoning', ['replay']],
			['tools', 'Tools', ['tool_choice', 'vision']],
			['limits', 'Limits', ['context_window']]
		]);
	});

	it('carries value, source and every layer of a field', () => {
		expect(rowOf()).toMatchObject({
			label: 'Context window',
			kind: 'int',
			min: 1024,
			value: 65536,
			source: 'discovered',
			seed: 32768,
			discovered: 65536,
			savedOverride: null,
			edited: false,
			editable: true
		});
		expect(rowOf({}, 'replay')).toMatchObject({ label: 'Reasoning replay', source: 'override', savedOverride: 'preserve' });
	});

	it('puts a field the schema does not know last, read-only', () => {
		const profile: LLMProfile = {
			...PROFILE,
			fields: { ...PROFILE.fields, prices: { value: { input: 3 }, source: 'seed', seed: { input: 3 }, discovered: null, override: null } }
		};
		const groups = buildProfileGroups(profile, SCHEMA, {});
		const last = groups.at(-1)!;

		expect(last.key).toBe('other');
		expect(last.rows[0]).toMatchObject({ name: 'prices', kind: 'unknown', editable: false });
	});

	it('skips a schema field the profile has no value for', () => {
		const profile: LLMProfile = { ...PROFILE, fields: { replay: PROFILE.fields.replay } };

		expect(buildProfileGroups(profile, SCHEMA, {}).flatMap((group) => group.rows.map((row) => row.name))).toEqual(['replay']);
	});

	it('shows a draft value as an override on top of the saved layers', () => {
		const row = rowOf({ context_window: 131072 });

		expect(row).toMatchObject({ value: 131072, source: 'override', edited: true, savedValue: 65536 });
	});

	it('falls back to the discovered value, then the seed, when the draft clears the override', () => {
		expect(rowOf({ replay: null }, 'replay')).toMatchObject({ value: 'send_all', source: 'seed', edited: true });
		const profile: LLMProfile = {
			...PROFILE,
			fields: { replay: { value: 'preserve', source: 'override', seed: 'send_all', discovered: 'drop_earlier', override: 'preserve' } }
		};
		const row = buildProfileGroups(profile, SCHEMA, { replay: null })[0].rows[0];
		expect(row).toMatchObject({ value: 'drop_earlier', source: 'discovered' });
	});

	it('flags a number below the schema minimum and not before it is edited', () => {
		expect(rowOf({ context_window: 512 }).problem).toBe('Minimum is 1,024');
		expect(rowOf({}).problem).toBeNull();
		expect(rowOf({ context_window: 1500.5 }).problem).toBe('Enter a whole number');
	});
});

describe('setProfileDraft', () => {
	it('records a changed value', () => {
		expect(setProfileDraft({}, rowOf(), 131072)).toEqual({ context_window: 131072 });
	});

	it('drops the entry when the value goes back to what is saved', () => {
		const row = rowOf();
		const edited = setProfileDraft({}, row, 131072);

		expect(setProfileDraft(edited, row, 65536)).toEqual({});
	});

	it('compares sets without regard to order', () => {
		const row = rowOf({}, 'tool_choice');

		expect(setProfileDraft({}, row, ['none', 'auto'])).toEqual({});
		expect(setProfileDraft({}, row, ['auto'])).toEqual({ tool_choice: ['auto'] });
	});

	it('keeps other fields of the draft', () => {
		expect(setProfileDraft({ replay: 'never' }, rowOf(), 2048)).toEqual({ replay: 'never', context_window: 2048 });
	});

	it('does not mutate the draft it is given', () => {
		const draft = { replay: 'never' };
		setProfileDraft(draft, rowOf(), 2048);

		expect(draft).toEqual({ replay: 'never' });
	});
});

describe('clearProfileDraft', () => {
	it('asks the server to clear a saved override with null', () => {
		expect(clearProfileDraft({}, rowOf({}, 'replay'))).toEqual({ replay: null });
	});

	it('just forgets an unsaved edit on a field with no saved override', () => {
		const row = rowOf();

		expect(clearProfileDraft({ context_window: 2048 }, row)).toEqual({});
	});
});

describe('draft helpers', () => {
	it('counts and sends exactly the changed fields', () => {
		const draft = { replay: null, context_window: 2048 };

		expect(profileDraftCount(draft)).toBe(2);
		expect(profileDraftPayload(draft)).toEqual({ replay: null, context_window: 2048 });
		expect(profileDraftCount({})).toBe(0);
	});

	it('reports a problem only while an edited value is invalid', () => {
		expect(profileHasProblems(buildProfileGroups(PROFILE, SCHEMA, { context_window: 100 }))).toBe(true);
		expect(profileHasProblems(buildProfileGroups(PROFILE, SCHEMA, { context_window: 4096 }))).toBe(false);
	});
});

describe('profileLayers', () => {
	it('names the layers the effective value did not come from', () => {
		expect(profileLayers(rowOf())).toEqual(['Seed 32,768']);
		expect(profileLayers(rowOf({ context_window: 4096 }))).toEqual(['Seed 32,768', 'Discovered 65,536']);
		expect(profileLayers(rowOf({}, 'vision'))).toEqual([]);
	});
});

describe('formatProfileValue', () => {
	it('prints each kind of value readably', () => {
		expect(formatProfileValue(null)).toBe('—');
		expect(formatProfileValue(true)).toBe('Yes');
		expect(formatProfileValue(65536)).toBe('65,536');
		expect(formatProfileValue(['a', 'b'])).toBe('a, b');
		expect(formatProfileValue([])).toBe('None');
		expect(formatProfileValue({ input: 3 })).toBe('{"input":3}');
		expect(formatProfileValue('preserve')).toBe('preserve');
	});
});

describe('sameProfileValue', () => {
	it('treats null and undefined alike and sets as unordered', () => {
		expect(sameProfileValue(null, undefined)).toBe(true);
		expect(sameProfileValue(['b', 'a'], ['a', 'b'])).toBe(true);
		expect(sameProfileValue(1, 2)).toBe(false);
	});
});
