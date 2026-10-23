export const MAX_NUMBER_GRID = 200;
export const MAX_STEP_PLACES = 6;
export const MAX_NUMBER_EXPONENT = 12;
export const MAX_NUMBER_DIGITS = 12;

export interface DecimalValue {
	coef: bigint;
	exp: number;
	digits: number;
	negative: boolean;
}

export interface NumberGrid {
	places: number;
	count: number;
	minimum: DecimalValue;
	maximum: DecimalValue;
	step: DecimalValue;
}

export type NumberGridIssue = 'range' | 'step' | 'precision' | 'value' | 'too-large';

export type NumberGridResult = { grid: NumberGrid; issue: null } | { grid: null; issue: NumberGridIssue; count?: number };

export interface NumberGridInput {
	min: unknown;
	max: unknown;
	step: unknown;
	value?: unknown;
}

const DECIMAL_PATTERN = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/;

export function parseDecimal(raw: unknown): DecimalValue | null {
	if (typeof raw !== 'number' && typeof raw !== 'string') return null;
	const text = String(raw).trim();
	const match = DECIMAL_PATTERN.exec(text);
	if (!match) return null;
	const [, sign, whole, fraction = '', exponent = '0'] = match;
	if (whole === '' && fraction === '') return null;
	const stripped = (whole + fraction).replace(/^0+/, '');
	const digitsText = stripped === '' ? '0' : stripped;
	const exp = Number(exponent) - fraction.length;
	const digits = digitsText.length;
	if (exp + digits - 1 >= MAX_NUMBER_EXPONENT || digits > MAX_NUMBER_DIGITS) return null;
	const coef = BigInt(digitsText) * (sign === '-' ? -1n : 1n);
	return { coef, exp, digits, negative: sign === '-' };
}

function places(value: DecimalValue): number {
	return Math.max(0, -value.exp);
}

function pow10(n: number): bigint {
	return 10n ** BigInt(n);
}

function atExponent(value: DecimalValue, exp: number): bigint {
	return value.coef * pow10(value.exp - exp);
}

function roundHalfUpDiv(numerator: bigint, divisor: bigint): bigint {
	const abs = numerator < 0n ? -numerator : numerator;
	const quotient = (abs * 2n + divisor) / (divisor * 2n);
	return numerator < 0n ? -quotient : quotient;
}

function formatUnits(units: bigint, decimals: number, negative: boolean): string {
	const abs = units < 0n ? -units : units;
	const digits = abs.toString().padStart(decimals + 1, '0');
	const whole = digits.slice(0, digits.length - decimals);
	const fraction = decimals > 0 ? `.${digits.slice(digits.length - decimals)}` : '';
	return `${negative || units < 0n ? '-' : ''}${whole}${fraction}`;
}

export function parseNumberGrid(input: NumberGridInput): NumberGridResult {
	const minimum = parseDecimal(input.min);
	const maximum = parseDecimal(input.max);
	if (!minimum || !maximum) return { grid: null, issue: 'range' };
	const step = parseDecimal(input.step);
	if (!step || step.coef <= 0n) return { grid: null, issue: 'step' };
	const exp = Math.min(minimum.exp, maximum.exp, step.exp);
	const low = atExponent(minimum, exp);
	const high = atExponent(maximum, exp);
	if (low > high) return { grid: null, issue: 'range' };
	const stepPlaces = places(step);
	if (stepPlaces > MAX_STEP_PLACES || places(minimum) > stepPlaces) return { grid: null, issue: 'precision' };
	if (input.value !== undefined && input.value !== null && !parseDecimal(input.value)) {
		return { grid: null, issue: 'value' };
	}
	const count = Number((high - low) / atExponent(step, exp)) + 1;
	if (count > MAX_NUMBER_GRID) return { grid: null, issue: 'too-large', count };
	return { grid: { places: stepPlaces, count, minimum, maximum, step }, issue: null };
}

export function formatGridIndex(grid: NumberGrid, index: number): string {
	const base = atExponent(grid.minimum, -grid.places);
	const stride = atExponent(grid.step, -grid.places);
	return formatUnits(base + stride * BigInt(index), grid.places, false);
}

export function gridValues(grid: NumberGrid): string[] {
	return Array.from({ length: grid.count }, (_, index) => formatGridIndex(grid, index));
}

export function clampFormat(grid: NumberGrid, raw: unknown): string {
	const parsed = raw === undefined || raw === null ? grid.minimum : parseDecimal(raw);
	const value = parsed ?? grid.minimum;
	const exp = Math.min(value.exp, grid.minimum.exp, grid.maximum.exp);
	const clamped =
		atExponent(value, exp) < atExponent(grid.minimum, exp)
			? grid.minimum
			: atExponent(value, exp) > atExponent(grid.maximum, exp)
				? grid.maximum
				: value;
	const shift = clamped.exp + grid.places;
	const units = shift >= 0 ? clamped.coef * pow10(shift) : roundHalfUpDiv(clamped.coef, pow10(-shift));
	return formatUnits(units, grid.places, clamped.negative);
}

export function gridWire(grid: NumberGrid): string {
	if (grid.count === 1) return formatGridIndex(grid, 0);
	return `{${gridValues(grid).join('|')}}`;
}
