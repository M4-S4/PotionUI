import { describe, it, expect, vi, afterEach } from 'vitest';
import { STATUS_LINES, createStatusLineRotator, laterPhase, phaseFor, pickLine, STATUS_LINE_MS, STATUS_FADE_MS } from './statusLines';

function seeded(seed: number) {
	let state = seed | 0;
	return () => {
		state = (state + 0x6d2b79f5) | 0;
		let t = Math.imul(state ^ (state >>> 15), state | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function harness(reduce = false) {
	const lines: string[] = [];
	const fades: boolean[] = [];
	const rotator = createStatusLineRotator({
		onLine: (line) => lines.push(line),
		onFade: (fading) => fades.push(fading),
		reducedMotion: () => reduce,
		random: seeded(7)
	});
	return { lines, fades, rotator };
}

afterEach(() => {
	vi.useRealTimers();
});

describe('status line lists', () => {
	const all = Object.values(STATUS_LINES).flat();

	it('has about twenty run lines and a small loading set', () => {
		expect(STATUS_LINES.running.length + STATUS_LINES.finishing.length).toBeGreaterThanOrEqual(20);
		expect(STATUS_LINES.loading.length).toBeGreaterThanOrEqual(3);
	});

	it('keeps every line at 28 characters or fewer, unique and free of error words', () => {
		expect(all.filter((line) => [...line].length > 28)).toEqual([]);
		expect(new Set(all).size).toBe(all.length);
		expect(all.filter((line) => /error|fail|crash|broke|stuck|oops|wrong|spill|explod/i.test(line))).toEqual([]);
	});
});

describe('phaseFor and laterPhase', () => {
	it('maps a missing fraction to loading and the last fifth to finishing', () => {
		expect(phaseFor(null, false)).toBe('loading');
		expect(phaseFor(undefined, false)).toBe('loading');
		expect(phaseFor(0, false)).toBe('running');
		expect(phaseFor(0.79, false)).toBe('running');
		expect(phaseFor(0.8, false)).toBe('finishing');
	});

	it('never switches to finishing under reduced motion', () => {
		expect(phaseFor(0.95, true)).toBe('running');
	});

	it('only moves forward within a run', () => {
		expect(laterPhase('finishing', 'running')).toBe('finishing');
		expect(laterPhase('running', 'loading')).toBe('running');
		expect(laterPhase('loading', 'running')).toBe('running');
		expect(laterPhase(null, 'loading')).toBe('loading');
	});
});

describe('pickLine', () => {
	it('never returns the current line', () => {
		const random = seeded(3);
		let current = '';
		for (let i = 0; i < 200; i++) {
			const next = pickLine(STATUS_LINES.running, current, random);
			expect(next).not.toBe(current);
			expect(STATUS_LINES.running).toContain(next);
			current = next;
		}
	});
});

describe('createStatusLineRotator', () => {
	it('shows a line at once, then crossfades to a different one every interval', () => {
		vi.useFakeTimers();
		const { lines, fades, rotator } = harness();
		rotator.sync('running', 'run-1');
		expect(lines).toHaveLength(1);
		for (let i = 0; i < 6; i++) {
			vi.advanceTimersByTime(i === 0 ? STATUS_LINE_MS : STATUS_LINE_MS - STATUS_FADE_MS);
			expect(fades.at(-1)).toBe(true);
			vi.advanceTimersByTime(STATUS_FADE_MS);
			expect(fades.at(-1)).toBe(false);
		}
		expect(lines).toHaveLength(7);
		lines.slice(1).forEach((line, index) => expect(line).not.toBe(lines[index]));
		expect(lines.every((line) => STATUS_LINES.running.includes(line))).toBe(true);
		rotator.dispose();
	});

	it('crossfades straight to a finishing line and never goes back within the run', () => {
		vi.useFakeTimers();
		const { lines, rotator } = harness();
		rotator.sync('running', 'run-1');
		rotator.sync('finishing', 'run-1');
		vi.advanceTimersByTime(STATUS_FADE_MS);
		expect(STATUS_LINES.finishing).toContain(lines.at(-1));
		const count = lines.length;
		rotator.sync('running', 'run-1');
		expect(lines).toHaveLength(count);
		rotator.dispose();
	});

	it('keeps one line for the whole run under reduced motion', () => {
		vi.useFakeTimers();
		const { lines, fades, rotator } = harness(true);
		rotator.sync('loading', 'run-1');
		vi.advanceTimersByTime(STATUS_LINE_MS * 5);
		expect(lines).toHaveLength(1);
		expect(STATUS_LINES.loading).toContain(lines[0]);
		rotator.sync('running', 'run-1');
		expect(lines).toHaveLength(2);
		expect(STATUS_LINES.running).toContain(lines[1]);
		vi.advanceTimersByTime(STATUS_LINE_MS * 5);
		expect(lines).toHaveLength(2);
		expect(fades.every((fading) => fading === false)).toBe(true);
		rotator.dispose();
	});

	it('stops rotating when reduced motion turns on mid-run and resumes when it turns off', () => {
		vi.useFakeTimers();
		let reduce = false;
		const lines: string[] = [];
		const rotator = createStatusLineRotator({ onLine: (line) => lines.push(line), onFade: () => {}, reducedMotion: () => reduce, random: seeded(5) });
		rotator.sync('running', 'run-1');
		reduce = true;
		rotator.sync('running', 'run-1');
		vi.advanceTimersByTime(STATUS_LINE_MS * 4);
		expect(lines).toHaveLength(1);
		reduce = false;
		rotator.sync('running', 'run-1');
		vi.advanceTimersByTime(STATUS_LINE_MS + STATUS_FADE_MS);
		expect(lines).toHaveLength(2);
		rotator.dispose();
	});

	it('stops when the run ends and starts fresh for the next run', () => {
		vi.useFakeTimers();
		const { lines, rotator } = harness();
		rotator.sync('running', 'run-1');
		rotator.sync(null, null);
		vi.advanceTimersByTime(STATUS_LINE_MS * 3);
		expect(lines).toHaveLength(1);
		rotator.sync('loading', 'run-2');
		expect(lines).toHaveLength(2);
		expect(STATUS_LINES.loading).toContain(lines[1]);
		rotator.dispose();
	});
});
