export type StatusPhase = 'loading' | 'running' | 'finishing';

export const STATUS_LINES: Record<StatusPhase, readonly string[]> = {
	running: [
		'Brewing…',
		'Stirring the cauldron…',
		'Distilling pixels…',
		'Adding a pinch of noise…',
		'Letting it simmer…',
		'Grinding moonstone…',
		'Measuring the drops…',
		'Swirling the flask…',
		'Steeping the palette…',
		'Whisking in the details…',
		'Reading the recipe…',
		'Thickening the shadows…',
		'Folding in the light…',
		'Coaxing out the shapes…',
		'Sifting the steam…',
		'Tuning the elixir…',
		'Watching it bubble…',
		'Balancing the mixture…'
	],
	finishing: ['Bottling the result…', 'Corking the bottle…', 'Adding the final drop…', 'Labeling the potion…'],
	loading: ['Warming the cauldron…', 'Lighting the burner…', 'Gathering ingredients…', 'Dusting off the recipe…']
};

export const STATUS_LINE_MS = 3500;
export const STATUS_FADE_MS = 250;
export const FINISHING_FROM = 0.8;

const ORDER: Record<StatusPhase, number> = { loading: 0, running: 1, finishing: 2 };

export function phaseFor(progress: number | null | undefined, reducedMotion: boolean): StatusPhase {
	if (progress === null || progress === undefined) return 'loading';
	return progress >= FINISHING_FROM && !reducedMotion ? 'finishing' : 'running';
}

export function laterPhase(current: StatusPhase | null, next: StatusPhase): StatusPhase {
	return current !== null && ORDER[current] > ORDER[next] ? current : next;
}

export function pickLine(pool: readonly string[], current: string, random: () => number = Math.random): string {
	const options = pool.filter((line) => line !== current);
	const list = options.length > 0 ? options : pool;
	return list[Math.min(list.length - 1, Math.floor(random() * list.length))];
}

export interface StatusLineRotator {
	sync(phase: StatusPhase | null, runKey: string | null): void;
	dispose(): void;
}

export function createStatusLineRotator(options: {
	onLine: (line: string) => void;
	onFade: (fading: boolean) => void;
	reducedMotion: () => boolean;
	random?: () => number;
	intervalMs?: number;
	fadeMs?: number;
}): StatusLineRotator {
	const random = options.random ?? Math.random;
	const intervalMs = options.intervalMs ?? STATUS_LINE_MS;
	const fadeMs = options.fadeMs ?? STATUS_FADE_MS;
	let phase: StatusPhase | null = null;
	let run: string | null = null;
	let line = '';
	let reduced = false;
	let timer: ReturnType<typeof setInterval> | undefined;
	let swap: ReturnType<typeof setTimeout> | undefined;

	function show(next: string, animate: boolean) {
		clearTimeout(swap);
		if (!animate) {
			line = next;
			options.onFade(false);
			options.onLine(next);
			return;
		}
		options.onFade(true);
		swap = setTimeout(() => {
			line = next;
			options.onLine(next);
			options.onFade(false);
		}, fadeMs);
	}

	function rotate() {
		if (phase) show(pickLine(STATUS_LINES[phase], line, random), true);
	}

	function stop() {
		clearInterval(timer);
		clearTimeout(swap);
		timer = undefined;
		swap = undefined;
		options.onFade(false);
	}

	return {
		sync(nextPhase, runKey) {
			if (nextPhase === null) {
				if (phase !== null) stop();
				phase = null;
				run = null;
				return;
			}
			const sameRun = phase !== null && runKey === run;
			const target = sameRun ? laterPhase(phase, nextPhase) : nextPhase;
			const reduce = options.reducedMotion();
			if (sameRun && target === phase) {
				if (reduce === reduced) return;
				reduced = reduce;
				stop();
				if (!reduce) timer = setInterval(rotate, intervalMs);
				return;
			}
			stop();
			phase = target;
			run = runKey;
			reduced = reduce;
			if (sameRun && !reduce) rotate();
			else show(pickLine(STATUS_LINES[target], line, random), false);
			if (!reduce) timer = setInterval(rotate, intervalMs);
		},
		dispose() {
			stop();
			phase = null;
			run = null;
		}
	};
}
