import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { activeTab, tabsStore } from '$lib/stores/tabs';
import type { Segment } from '$lib/types/segments';
import type { VariablesMap } from '$lib/utils/variableDefs';
import { libraryTabVariables, libraryVariableScope } from './libraryTabVariables';

const mood: VariablesMap = {
	mood: { type: 'choice', mode: 'pin', pinnedIndex: 0, options: ['calm', 'tense', 'wild'] }
};

const segment = (content: string): Segment => ({ id: content, content, chips: {}, enabled: true });

function setTabVariables(next: VariablesMap) {
	tabsStore.updateTab(get(activeTab).id, { variables: next });
}

beforeEach(() => setTabVariables({}));

afterEach(() => {
	tabsStore.updateTab(get(activeTab).id, { variables: {}, promptSegments: [], negativePromptSegments: [] });
});

describe('libraryTabVariables', () => {
	it('exposes the active tab variables', () => {
		setTabVariables(mood);
		expect(get(libraryTabVariables).variables).toEqual(mood);
	});

	it('setDef adds one definition to the active tab and keeps the others', () => {
		setTabVariables(mood);
		get(libraryTabVariables).setDef('age', { type: 'text', value: 'old' });
		expect(get(activeTab).variables).toEqual({ ...mood, age: { type: 'text', value: 'old' } });
		expect(get(libraryTabVariables).variables).toEqual(get(activeTab).variables);
	});

	it('setAll replaces the active tab variables', () => {
		setTabVariables(mood);
		get(libraryTabVariables).setAll({ age: { type: 'text', value: 'old' } });
		expect(get(activeTab).variables).toEqual({ age: { type: 'text', value: 'old' } });
	});

	it('writes to the tab that is active when called, not the one it was read from', () => {
		const first = get(activeTab).id;
		const source = get(libraryTabVariables);
		tabsStore.addTab();
		const second = get(activeTab).id;
		expect(second).not.toBe(first);
		source.setDef('age', { type: 'text', value: 'old' });
		expect(get(activeTab).variables).toEqual({ age: { type: 'text', value: 'old' } });
		tabsStore.setActiveTab(first);
		expect(get(activeTab).variables).toEqual({});
	});

	it('exposes the positive and negative prompt texts of the active tab', () => {
		tabsStore.updateTab(get(activeTab).id, {
			promptSegments: [segment('one'), segment('two')],
			negativePromptSegments: [segment('three')]
		});
		expect(get(libraryTabVariables).texts).toEqual(['one', 'two', 'three']);
	});

	describe('live subscription', () => {
		it('notifies once per relevant change and ignores unrelated tab fields', () => {
			const first = get(activeTab).id;
			const spy = vi.fn();
			const stop = libraryTabVariables.subscribe(spy);
			expect(spy).toHaveBeenCalledTimes(1);
			spy.mockClear();

			setTabVariables(mood);
			expect(spy).toHaveBeenCalledTimes(1);
			expect(spy.mock.calls[0][0].variables).toEqual(mood);
			spy.mockClear();

			tabsStore.updateTab(first, { name: 'renamed' });
			expect(spy).not.toHaveBeenCalled();

			tabsStore.updateTab(first, { promptSegments: [segment('one')] });
			expect(spy).toHaveBeenCalledTimes(1);
			expect(spy.mock.calls[0][0].texts).toEqual(['one']);
			spy.mockClear();

			tabsStore.addTab();
			expect(spy).toHaveBeenCalledTimes(1);
			expect(spy.mock.calls[0][0].variables).toEqual({});
			spy.mockClear();

			tabsStore.setActiveTab(first);
			expect(spy).toHaveBeenCalledTimes(1);
			expect(spy.mock.calls[0][0].variables).toEqual(mood);

			stop();
		});
	});
});

describe('libraryVariableScope', () => {
	it('counts usages and condition reads across the tab prompts and the draft', () => {
		tabsStore.updateTab(get(activeTab).id, {
			promptSegments: [segment('a ${mood} <<if $mood = "calm">>x<<end>>')],
			negativePromptSegments: [segment('${mood} bad')]
		});

		const scope = libraryVariableScope(get(libraryTabVariables), ['and ${mood}']);

		expect(scope.usageCounts).toEqual({ mood: 3 });
		expect(Object.keys(scope.readers)).toEqual(['mood']);
	});

	it('covers the draft alone when the tab has no prompt text', () => {
		const scope = libraryVariableScope(get(libraryTabVariables), ['one ${a} <<if $b = "x">>y<<end>>']);

		expect(scope.usageCounts).toEqual({ a: 1 });
		expect(Object.keys(scope.readers)).toEqual(['b']);
	});
});
