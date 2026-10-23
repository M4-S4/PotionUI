import { describe, it, expect } from 'vitest';
import { buildChatFormState } from './chatFormState';

const base = {
	generationPresetActive: true,
	sessionId: null,
	videoDirector: { active: false, doc: null, capabilities: null },
	musicDirector: { active: false, doc: null, capabilities: null }
};

const tab = {
	id: 'tab-1',
	selectedPreset: 'native/SDXL/realistic',
	selectedMode: 'txt2img',
	selectedVariant: null,
	formData: { steps: 20 },
	variableRolls: { mood: { optionIndex: 1, value: 'sunlit', rolledAt: 1 } }
};

describe('buildChatFormState', () => {
	it('carries the active tab id and never the variables themselves', () => {
		const state = buildChatFormState({ ...base, tab, generatePageActive: true });
		expect(state.tab_id).toBe('tab-1');
		expect(state).not.toHaveProperty('variables');
	});

	it('sends each rolled variable as its last-roll text and omits names without a roll', () => {
		const withRolls = {
			...tab,
			variableRolls: {
				mood: { optionIndex: 1, value: 'sunlit', rolledAt: 1 },
				pose: {
					optionIndex: 0,
					value: 'popping',
					rolledAt: 1,
					because: { var: 'music', value: 'hip hop' },
					eligible: 2,
					total: 5
				},
				age: { optionIndex: 3, value: '21', rolledAt: 1 },
				angle: { optionIndex: 0, value: 'low', rolledAt: 1, eligible: 1, total: 4 },
				blank: { optionIndex: 0, value: '  ', rolledAt: 1 }
			}
		};
		const state = buildChatFormState({ ...base, tab: withRolls, generatePageActive: true });
		expect(state.variable_rolls).toEqual({
			mood: 'sunlit',
			pose: 'popping (because $music = hip hop, 2 of 5 eligible)',
			age: '21',
			angle: 'low (1 of 4 eligible)'
		});
	});

	it('sends no rolls when the tab has none', () => {
		expect(buildChatFormState({ ...base, tab: { id: 't2' }, generatePageActive: true }).variable_rolls).toEqual({});
		expect(buildChatFormState({ ...base, tab: { id: 't2', variableRolls: {} }, generatePageActive: true }).variable_rolls).toEqual({});
	});

	it('withholds tab id and form data away from the Generate page', () => {
		const state = buildChatFormState({ ...base, tab, generatePageActive: false });
		expect(state.tab_id).toBeNull();
		expect(state.form_data).toEqual({});
	});

	it('reports a null tab id when there is no tab', () => {
		expect(buildChatFormState({ ...base, tab: null, generatePageActive: true }).tab_id).toBeNull();
	});
});
