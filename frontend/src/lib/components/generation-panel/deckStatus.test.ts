import { describe, it, expect } from 'vitest';
import { deckEyebrow, deckTitle, deriveDeckState, shouldAnnounceQuarter } from './deckStatus';

const base = { isGenerating: false, hasProgressFraction: false, otherTabName: null, disabled: false, resultStatus: null };

describe('deriveDeckState', () => {
	it('puts a run in flight first, split by whether progress has a fraction', () => {
		expect(deriveDeckState({ ...base, isGenerating: true, hasProgressFraction: true, disabled: true })).toBe('running');
		expect(deriveDeckState({ ...base, isGenerating: true })).toBe('loading');
	});

	it('orders other tab, blocked, failed, done and ready', () => {
		expect(deriveDeckState({ ...base, otherTabName: 'Short film', disabled: true })).toBe('other');
		expect(deriveDeckState({ ...base, disabled: true, resultStatus: 'failed' })).toBe('blocked');
		expect(deriveDeckState({ ...base, resultStatus: 'failed' })).toBe('failed');
		expect(deriveDeckState({ ...base, resultStatus: 'completed' })).toBe('done');
		expect(deriveDeckState(base)).toBe('ready');
	});
});

describe('deckTitle', () => {
	const input = { otherTabName: null, compareArmed: false, stopping: false, hasResult: true, playful: true };

	it('keeps plain, literal titles for every state', () => {
		expect(deckTitle({ ...input, state: 'running' }).literal).toBe('Generating');
		expect(deckTitle({ ...input, state: 'running', stopping: true }).literal).toBe('Finishing this image');
		expect(deckTitle({ ...input, state: 'loading' }).literal).toBe('Loading model');
		expect(deckTitle({ ...input, state: 'other', otherTabName: 'Short film' }).literal).toBe('Generating in Short film');
		expect(deckTitle({ ...input, state: 'blocked' }).literal).toBe("Can't generate yet");
		expect(deckTitle({ ...input, state: 'blocked', compareArmed: true }).literal).toBe("Can't compare yet");
		expect(deckTitle({ ...input, state: 'failed' }).visible).toBe('Failed');
		expect(deckTitle({ ...input, state: 'done' }).visible).toBe('Done');
		expect(deckTitle({ ...input, state: 'ready' }).visible).toBe('Ready to generate');
		expect(deckTitle({ ...input, state: 'ready', compareArmed: true }).visible).toBe('Ready to compare');
	});

	it('says Ready to brew only for a fresh tab with the playful lines on', () => {
		expect(deckTitle({ ...input, state: 'ready', hasResult: false })).toEqual({ literal: 'Ready to generate', visible: 'Ready to brew' });
		expect(deckTitle({ ...input, state: 'ready', hasResult: false, playful: false }).visible).toBe('Ready to generate');
		expect(deckTitle({ ...input, state: 'blocked', hasResult: false }).visible).toBe("Can't generate yet");
	});
});

describe('deckEyebrow', () => {
	it('names the thumbnail', () => {
		expect(deckEyebrow('running', true, '18.4s')).toBe('Now generating');
		expect(deckEyebrow('loading', false, 'none')).toBe('Now generating');
		expect(deckEyebrow('done', true, '18.4s')).toBe('New result · 18.4s');
		expect(deckEyebrow('ready', true, '18.4s')).toBe('Last result · 18.4s');
		expect(deckEyebrow('ready', true, 'none')).toBe('Last result');
		expect(deckEyebrow('blocked', false, 'none')).toBe('No results yet');
	});
});

describe('shouldAnnounceQuarter', () => {
	it('announces each quarter once, below 100%', () => {
		expect(shouldAnnounceQuarter(10, 0)).toBeNull();
		expect(shouldAnnounceQuarter(26, 0)).toBe(25);
		expect(shouldAnnounceQuarter(49, 25)).toBeNull();
		expect(shouldAnnounceQuarter(80, 25)).toBe(75);
		expect(shouldAnnounceQuarter(100, 75)).toBeNull();
		expect(shouldAnnounceQuarter(30, 75)).toBeNull();
	});
});
