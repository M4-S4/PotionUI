import { describe, it, expect } from 'vitest';
import { buildGenerationRequest, prepareRequest, prepareRequestFromSession, shuffleSegmentChips } from './prepareRequest';

function sequence(...values: number[]) {
	let index = 0;
	return () => values[index++ % values.length];
}

function baseTab(overrides: Record<string, unknown> = {}) {
	return {
		selectedPreset: 'preset-1',
		selectedMode: 'txt2img',
		selectedVariant: null,
		formData: { seed: 7, quantity: 1 },
		prompt: '  a lighthouse at dusk  ',
		negativePrompt: ' blur ',
		promptSegments: [],
		negativePromptSegments: [],
		promptTabs: undefined,
		activePromptTab: 0,
		promptRelay: undefined,
		videoDirector: undefined,
		musicDirector: undefined,
		variables: undefined,
		autoTagIds: [],
		autoCollectionIds: undefined,
		sourcePromptId: null,
		directorRuns: undefined,
		...overrides
	} as any;
}

function prepare(tab: any, extra: Record<string, unknown> = {}) {
	return prepareRequest({
		tab,
		tabId: 'tab-1',
		canBatch: false,
		segmentJoin: 'space',
		promptRelayActive: false,
		promptlessActive: false,
		videoDirector: null,
		musicDirector: null,
		random: () => 0,
		now: () => 1000,
		...extra
	} as any);
}

function batchTab({ active }: { active: number }) {
	const entries = [
		{ prompt: 'first', negativePrompt: 'neg one', promptSegments: [], negativePromptSegments: [] },
		{ prompt: 'unused', negativePrompt: '', promptSegments: [{ id: 'x', content: 'second from segments' }], negativePromptSegments: [] },
		{ prompt: 'third', negativePrompt: '', promptSegments: [], negativePromptSegments: [] }
	];
	const current = entries[active];
	return baseTab({
		promptTabs: entries,
		activePromptTab: active,
		prompt: current.prompt,
		negativePrompt: current.negativePrompt,
		promptSegments: current.promptSegments,
		negativePromptSegments: current.negativePromptSegments
	});
}

const chip = {
	id: 'c1',
	categoryPath: 'style',
	valueId: 'v1',
	label: 'Oil',
	value: 'oil painting',
	allValues: [
		{ id: 'v1', label: 'Oil', value: 'oil painting' },
		{ id: 'v2', label: 'Ink', value: 'ink drawing' }
	],
	shuffle: true,
	autoRegen: false
};

describe('prepareRequest', () => {
	it('builds a plain prompt request with trimmed prompts and the full prompt state', () => {
		const result = prepare(baseTab({ autoTagIds: ['tag-1'], sourcePromptId: 'prompt-9', selectedVariant: 'custom' }));

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request).toEqual({
			preset_id: 'preset-1',
			prompts: [{ positive: 'a lighthouse at dusk', negative: 'blur' }],
			mode: 'txt2img',
			form_name: 'custom',
			form_data: { seed: 7, quantity: 1 },
			tag_ids: ['tag-1'],
			collection_ids: undefined,
			variables: undefined,
			tab_id: 'tab-1',
			source_prompt_id: 'prompt-9',
			prompt_state: {
				prompt: '  a lighthouse at dusk  ',
				negativePrompt: ' blur ',
				promptSegments: [],
				negativePromptSegments: [],
				variables: undefined,
				promptTabs: undefined,
				activePromptTab: 0,
				promptRelay: undefined,
				videoDirector: undefined
			},
			segments: []
		});
		expect(Object.keys(result.request)).toEqual([
			'preset_id', 'prompts', 'mode', 'form_name', 'form_data', 'tag_ids', 'collection_ids',
			'variables', 'tab_id', 'source_prompt_id', 'prompt_state', 'segments'
		]);
		expect(result.submittedPromptTemplate).toEqual({ positive: 'a lighthouse at dusk', negative: 'blur' });
		expect(result.shuffled.changed).toBe(false);
	});

	it('resolves segments, shuffles chips once and reports the shuffled segments', () => {
		const tab = baseTab({
			promptSegments: [{ id: 's1', content: 'a portrait in #style', chips: { c1: chip } }],
			negativePromptSegments: [{ id: 'n1', content: 'lowres' }]
		});

		const result = prepare(tab, { segmentJoin: 'paragraph' });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.prompts).toEqual([{ positive: 'a portrait in ink drawing', negative: 'lowres' }]);
		expect(result.shuffled.changed).toBe(true);
		expect(result.shuffled.promptSegments[0].chips!.c1.valueId).toBe('v2');
		expect(tab.promptSegments[0].chips.c1.valueId).toBe('v1');
		expect(result.request.segments).toHaveLength(2);
		expect(result.request.prompt_state!.promptSegments).toBe(tab.promptSegments);
	});

	it('sends a single prompt without a batch flag', () => {
		const result = prepare(baseTab());

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.prompts).toEqual([{ positive: 'a lighthouse at dusk', negative: 'blur' }]);
		expect('batch_prompts' in result.request).toBe(false);
		expect('prompts' in result.request.prompt_state!).toBe(false);
	});

	it('sends every prompt with the batch flag when the preset can batch', () => {
		const tab = batchTab({ active: 1 });

		const result = prepare(tab, { canBatch: true });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.batch_prompts).toBe(true);
		expect(result.request.prompts).toEqual([
			{ positive: 'first', negative: 'neg one' },
			{ positive: 'second from segments', negative: '' },
			{ positive: 'third', negative: '' }
		]);
		expect(result.request.segments).toEqual([
			expect.objectContaining({ channel: 'positive', prompt_index: 1, text: 'second from segments' })
		]);
		const state = result.request.prompt_state as Record<string, any>;
		expect(state.prompts).toHaveLength(3);
		expect(state.prompts[1]).toEqual({
			prompt: 'unused',
			negativePrompt: '',
			promptSegments: [{ id: 'x', content: 'second from segments' }],
			negativePromptSegments: []
		});
		expect(state.prompt).toBe('first');
		expect('promptTabs' in state).toBe(false);
	});

	it('sends only the first prompt when the preset cannot batch', () => {
		const result = prepare(batchTab({ active: 2 }), { canBatch: false });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect('batch_prompts' in result.request).toBe(false);
		expect(result.request.prompts).toEqual([{ positive: 'first', negative: 'neg one' }]);
		expect(result.request.segments).toEqual([]);
		expect(result.shuffled.changed).toBe(false);
	});

	it('refuses a batch that contains an empty prompt', () => {
		const tab = batchTab({ active: 0 });
		tab.promptTabs[2] = { ...tab.promptTabs[2], prompt: '' };

		expect(prepare(tab, { canBatch: true })).toEqual({ ok: false, code: 'no_prompt', reason: null });
	});

	it('takes prompts and timing from the prompt relay timeline', () => {
		const relay = {
			global_prompt: '  open sea ',
			timeline: { duration: 5, fps: 24, segments: [{ start: 2, end: 4, text: ' storm ' }, { start: 0, end: 2, text: 'calm' }] }
		};

		const result = prepare(baseTab({ promptRelay: relay }), { promptRelayActive: true });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.prompts).toEqual([{ positive: 'open sea | calm | storm', negative: '' }]);
		expect(result.request.form_data).toEqual({ seed: 7, quantity: 1, global_prompt: 'open sea', timeline: relay.timeline });
	});

	it('rolls shuffle variables into the request and returns the rolls', () => {
		const variables = {
			mood: { type: 'choice', mode: 'shuffle', pinnedIndex: null, options: ['calm', 'stormy'] },
			place: { type: 'text', value: 'harbour' }
		};

		const result = prepare(baseTab({ variables, prompt: 'a ${mood} ${place}' }), { random: sequence(0.9), now: () => 5 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.variables).toEqual({ mood: 'stormy', place: 'harbour' });
		expect(result.variableRolls.mood).toMatchObject({ optionIndex: 1, value: 'stormy', rolledAt: 5 });
		expect(result.request.prompt_state!.variables).toBe(variables);
	});

	it('rolls a shuffle number once when a condition reads it, and per usage otherwise', () => {
		const variables = { age: { type: 'number', mode: 'shuffle', min: 18, max: 80, step: 1, value: 18 } };
		const plain = prepare(baseTab({ variables, prompt: '${age} and ${age}' }), { random: sequence(0, 0.5) });
		expect(plain.ok && plain.request.variables).toEqual({ age: '18', 'age-2': '49' });

		const read = prepare(
			baseTab({ variables, prompt: '${age} and <<if $age > 65>>grey<<else>>dark<<end>> and ${age}' }),
			{ random: sequence(0, 0.5) }
		);
		expect(read.ok).toBe(true);
		if (!read.ok) return;
		expect(read.request.variables).toEqual({ age: '18' });
		expect(read.request.prompts?.[0].positive).toBe('${age} and <<if $age > 65>>grey<<else>>dark<<end>> and ${age}');
	});

	it('refuses silently without a preset or a prompt, unless the mode needs no prompt', () => {
		expect(prepare(baseTab({ selectedPreset: null }))).toEqual({ ok: false, code: 'no_preset', reason: null });
		expect(prepare(baseTab({ prompt: '   ' }))).toEqual({ ok: false, code: 'no_prompt', reason: null });
		const promptless = prepare(baseTab({ prompt: '' }), { promptlessActive: true });
		expect(promptless.ok).toBe(true);
	});
});

describe('buildGenerationRequest', () => {
	it('leaves variables out of the prompt state when asked, as a Director shot does', () => {
		const tab = baseTab({ variables: { place: { type: 'text', value: 'harbour' } } });

		const request = buildGenerationRequest(tab, {
			tabId: 'tab-1',
			prompts: [{ positive: 'shot one', negative: '' }],
			formData: { ...tab.formData, video_director: { mode: 'i2v' } },
			variables: { place: 'harbour' },
			promptStateVariables: false
		});

		expect(Object.keys(request.prompt_state!)).toEqual([
			'prompt', 'negativePrompt', 'promptSegments', 'negativePromptSegments',
			'promptTabs', 'activePromptTab', 'promptRelay', 'videoDirector'
		]);
		expect(request.form_data).toEqual({ seed: 7, quantity: 1, video_director: { mode: 'i2v' } });
		expect(request.variables).toEqual({ place: 'harbour' });
	});
});

describe('shuffleSegmentChips', () => {
	it('keeps segments without shuffle chips untouched', () => {
		const segments = [{ id: 's', content: 'x', chips: { c1: { ...chip, shuffle: false } } }] as any;

		const result = shuffleSegmentChips(segments, () => 0);

		expect(result.changed).toBe(false);
		expect(result.segments[0]).toBe(segments[0]);
	});
});

describe('prepareRequestFromSession', () => {
	it('reads the mode settings from preset vars and applies overrides on top of the session', () => {
		const result = prepareRequestFromSession({
			presetId: 'preset-1',
			mode: 'relay',
			session: {
				prompt: 'ignored by relay',
				formData: { seed: 1, steps: 8 },
				promptRelay: { global_prompt: 'harbour', timeline: { duration: 5, fps: 24, segments: [] } },
				selectedVariant: 'custom'
			},
			overrides: { formData: { steps: 24 } },
			presetVars: { prompt_relay_modes: ['relay'] },
			tabId: 'node-1',
			random: () => 0,
			now: () => 1
		});

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.request.prompts).toEqual([{ positive: 'harbour', negative: '' }]);
		expect(result.request.form_data).toMatchObject({ seed: 1, steps: 24, global_prompt: 'harbour' });
		expect(result.request.form_name).toBe('custom');
		expect(result.request.tab_id).toBe('node-1');
		expect(result.request.mode).toBe('relay');
	});

	it('treats a promptless mode as ready without a prompt', () => {
		const result = prepareRequestFromSession({
			presetId: 'preset-1',
			mode: 'upscale',
			session: { formData: { input_image: 'uploads/a.png' } },
			presetVars: { promptless_modes: ['upscale'] },
			random: () => 0,
			now: () => 1
		});

		expect(result.ok).toBe(true);
	});

	it('needs no prompt only while a conditional promptless entry holds', () => {
		const run = (guideOnly: boolean) =>
			prepareRequestFromSession({
				presetId: 'preset-1',
				mode: 'control',
				session: { prompt: '', formData: { guide: 'openpose', guide_only: guideOnly } },
				presetVars: {
					promptless_modes: [
						{
							mode: 'control',
							when: {
								logic: 'AND',
								conditions: [
									{ field: 'guide_only', equals: true },
									{ field: 'guide', not_in: ['none', 'grayscale'] }
								]
							}
						}
					]
				},
				random: () => 0,
				now: () => 1
			});

		const extractOnly = run(true);
		expect(extractOnly.ok).toBe(true);
		if (extractOnly.ok) expect(extractOnly.request.mode).toBe('control');

		const generating = run(false);
		expect(generating.ok).toBe(false);
		if (!generating.ok) expect(generating.code).toBe('no_prompt');
	});

	it('leaves the session and overrides untouched and shares nothing with them', () => {
		const freeze = (value: any): any => {
			if (value && typeof value === 'object') {
				Object.values(value).forEach(freeze);
				Object.freeze(value);
			}
			return value;
		};
		const session = freeze({
			prompt: 'a harbour',
			promptSegments: [{ id: 's1', content: 'a harbour at #style', chips: { c1: chip } }],
			formData: { seed: 1, steps: 8 }
		});
		const overrides = freeze({ formData: { steps: 24 }, negativePrompt: 'blur' });
		const before = JSON.stringify({ session, overrides });

		const result = prepareRequestFromSession({ presetId: 'preset-1', mode: 'txt2img', session, overrides, random: () => 0 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(JSON.stringify({ session, overrides })).toBe(before);
		expect(result.request.prompt_state!.promptSegments).not.toBe(session.promptSegments);
		expect(result.request.form_data).toEqual({ seed: 1, steps: 24 });
		expect(result.request.prompts![0].negative).toBe('blur');
		(result.request.prompt_state!.promptSegments as any[])[0].content = 'changed';
		expect(session.promptSegments[0].content).toBe('a harbour at #style');
	});

	it('follows the preset vars for how segments join and the capability for how many prompts', () => {
		const promptTab = (name: string) => ({
			prompt: '',
			negativePrompt: '',
			promptSegments: [{ id: `${name}1`, content: `${name} one` }, { id: `${name}2`, content: `${name} two` }],
			negativePromptSegments: []
		});
		const session = { promptTabs: [promptTab('a'), promptTab('b'), promptTab('c'), promptTab('d')], formData: {} };

		const tuned = prepareRequestFromSession({
			presetId: 'preset-1',
			mode: 'txt2img',
			session,
			presetVars: { prompt: { segment_join: 'paragraph' } },
			canBatch: true,
			random: () => 0
		});
		const plain = prepareRequestFromSession({ presetId: 'preset-1', mode: 'txt2img', session: { prompt: 'single', formData: {} }, random: () => 0 });

		expect(tuned.ok && plain.ok).toBe(true);
		if (!tuned.ok || !plain.ok) return;
		expect(tuned.request.prompts!.map((pair) => pair.positive)).toEqual(['a one\n\na two', 'b one\n\nb two', 'c one\n\nc two', 'd one\n\nd two']);
		expect(tuned.request.segments).toHaveLength(8);
		expect(plain.request.prompts).toEqual([{ positive: 'single', negative: '' }]);
	});

	it('reports a missing prompt for an ordinary mode', () => {
		const result = prepareRequestFromSession({ presetId: 'preset-1', mode: 'txt2img', session: {}, random: () => 0 });

		expect(result).toEqual({ ok: false, code: 'no_prompt', reason: null });
	});
});

describe('prepareRequest independent variable usages', () => {
	const mood = {
		type: 'choice',
		mode: 'shuffle',
		pinnedIndex: null,
		options: ['calm', 'wild', 'dark']
	};

	it('sends rewritten text and per-usage rolls but keeps the original text in segments and prompt_state', () => {
		const segments = [{ id: 's1', content: '${mood} then ${mood}' }];
		const tab = baseTab({
			prompt: '${mood} then ${mood}',
			negativePrompt: '${mood}',
			promptSegments: segments,
			variables: { mood }
		});
		const result = prepare(tab, { random: sequence(0, 0.5, 0.99) });
		if (!result.ok) throw new Error('expected ok');
		expect(result.request.prompts?.[0].positive).toBe('${mood} then ${mood-2}');
		expect(result.request.prompts?.[0].negative).toBe('${mood-3}');
		expect(result.request.variables).toEqual({ mood: 'calm', 'mood-2': 'wild', 'mood-3': 'dark' });
		expect(Object.keys(result.variableRolls).sort()).toEqual(['mood', 'mood-2', 'mood-3']);
		expect(result.request.prompt_state?.prompt).toBe('${mood} then ${mood}');
		expect(result.request.prompt_state?.promptSegments).toEqual(segments);
		expect(result.request.segments?.[0].text).toBe('${mood} then ${mood}');
	});

	it('leaves a linked variable untouched on the wire', () => {
		const tab = baseTab({ prompt: '${mood} ${mood}', variables: { mood: { ...mood, linked: true } } });
		const result = prepare(tab, { random: sequence(0.5) });
		if (!result.ok) throw new Error('expected ok');
		expect(result.request.prompts?.[0].positive).toBe('${mood} ${mood}');
		expect(result.request.variables).toEqual({ mood: 'wild' });
	});
});
