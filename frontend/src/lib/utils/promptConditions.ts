export const SENTINEL_OPEN = '';
export const SENTINEL_CLOSE = '';

export const MAX_SCAN_CHARS = 20_000;
export const MAX_BLOCKS = 128;
export const MAX_BRANCHES = 32;
export const MAX_HEAD_CHARS = 4096;
export const MAX_TESTS = 32;
export const MAX_VALUES = 12;
export const MAX_NUMBER_CHARS = 20;
export const MAX_STRING_CHARS = 2000;
export const MAX_NAME_CHARS = 60;
export const MAX_VALUE_NUMBER_CHARS = 40;
export const MAX_SCRUB_PASSES = 8;

export type ConditionJoiner = 'and' | 'or';
export type NumberOp = '=' | '!=' | '<' | '<=' | '>' | '>=';
export type StringOp = '=' | '!=' | 'contains' | 'not contains';
export type MembershipOp = 'in' | 'not in';
export type EmptyOp = 'is empty' | 'is not empty';
export type ConditionOp = NumberOp | StringOp | MembershipOp | EmptyOp;
export type BranchKind = 'if' | 'elseif' | 'else';
export type ConditionErrorKind =
	| 'malformed_tag'
	| 'nested'
	| 'stray'
	| 'limit'
	| 'unclosed'
	| 'bad_head';

interface TestBase {
	var: string;
	start: number;
	end: number;
}

export interface EmptyTest extends TestBase {
	kind: 'none';
	op: EmptyOp;
}

export interface NumberTest extends TestBase {
	kind: 'number';
	op: NumberOp;
	number: string;
}

export interface StringTest extends TestBase {
	kind: 'string';
	op: StringOp;
	strings: string[];
}

export interface StringsTest extends TestBase {
	kind: 'strings';
	op: MembershipOp;
	strings: string[];
}

export interface RangeTest extends TestBase {
	kind: 'range';
	op: MembershipOp;
	low: string;
	high: string;
}

export type ConditionTest = EmptyTest | NumberTest | StringTest | StringsTest | RangeTest;

export interface Condition {
	joiner: ConditionJoiner;
	tests: ConditionTest[];
	start: number;
	end: number;
}

export interface ConditionBranch {
	kind: BranchKind;
	cond: Condition | null;
	head: string;
	headStart: number;
	tagStart: number;
	tagEnd: number;
	bodyStart: number;
	bodyEnd: number;
	body: string;
}

export interface ConditionBlock {
	index: number;
	start: number;
	end: number;
	branches: ConditionBranch[];
}

export interface ConditionError {
	kind: ConditionErrorKind;
	offset: number;
}

export interface ParsedTemplate {
	source: string;
	skeleton: string;
	blocks: ConditionBlock[];
	errors: ConditionError[];
}

export type ConditionValues = Readonly<Record<string, string>>;

export interface RenderResult {
	text: string;
	errors: ConditionError[];
}

type TokKind = 'if' | 'elseif' | 'else' | 'end' | 'junk';

interface Tok {
	kind: TokKind;
	start: number;
	end: number;
	head: string;
	headStart: number;
}

class HeadError extends Error {}

const SENTINEL_STRIP = /[]/g;

function isWs(c: string): boolean {
	return c === ' ' || c === '\t' || c === '\r' || c === '\n';
}

function isDigit(c: string): boolean {
	return c >= '0' && c <= '9' && c.length === 1;
}

function isWord(c: string): boolean {
	return (
		c.length === 1 &&
		((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c === '_')
	);
}

function isPySpace(code: number): boolean {
	return (
		(code >= 0x09 && code <= 0x0d) ||
		(code >= 0x1c && code <= 0x20) ||
		code === 0x85 ||
		code === 0xa0 ||
		code === 0x1680 ||
		(code >= 0x2000 && code <= 0x200a) ||
		code === 0x2028 ||
		code === 0x2029 ||
		code === 0x202f ||
		code === 0x205f ||
		code === 0x3000
	);
}

function pyStripBounds(s: string): [number, number] {
	let lo = 0;
	let hi = s.length;
	while (lo < hi && isPySpace(s.charCodeAt(lo))) lo++;
	while (hi > lo && isPySpace(s.charCodeAt(hi - 1))) hi--;
	return [lo, hi];
}

function pyStrip(s: string): string {
	const [lo, hi] = pyStripBounds(s);
	return s.slice(lo, hi);
}

function isBlank(c: string): boolean {
	return c === ' ' || c === '\t';
}

function hasContent(s: string): boolean {
	for (let i = 0; i < s.length; i++) {
		if (!isBlank(s.charAt(i))) return true;
	}
	return false;
}

function trimBlankStart(s: string): string {
	let i = 0;
	while (i < s.length && isBlank(s.charAt(i))) i++;
	return s.slice(i);
}

function trimBlankEnd(s: string): string {
	let i = s.length;
	while (i > 0 && isBlank(s.charAt(i - 1))) i--;
	return s.slice(0, i);
}

function casefold(s: string): string {
	return s.toUpperCase().toLowerCase();
}

interface Dec {
	int: bigint;
	scale: number;
}

function toDec(text: string): Dec {
	const dot = text.indexOf('.');
	if (dot < 0) return { int: BigInt(text), scale: 0 };
	return { int: BigInt(text.slice(0, dot) + text.slice(dot + 1)), scale: text.length - dot - 1 };
}

function cmpDec(a: Dec, b: Dec): number {
	let x = a.int;
	let y = b.int;
	if (a.scale < b.scale) x *= BigInt(10) ** BigInt(b.scale - a.scale);
	else if (b.scale < a.scale) y *= BigInt(10) ** BigInt(a.scale - b.scale);
	return x < y ? -1 : x > y ? 1 : 0;
}

function numberEnd(s: string, pos: number): number {
	let p = pos;
	if (s.charAt(p) === '-') p++;
	const digitsStart = p;
	while (isDigit(s.charAt(p))) p++;
	if (p === digitsStart) return -1;
	if (s.charAt(p) === '.' && isDigit(s.charAt(p + 1))) {
		p++;
		while (isDigit(s.charAt(p))) p++;
	}
	return p;
}

function skipWs(head: string, pos: number): number {
	let p = pos;
	while (isWs(head.charAt(p))) p++;
	return p;
}

function needWs(head: string, pos: number): number {
	const p = skipWs(head, pos);
	if (p === pos) throw new HeadError();
	return p;
}

function readNumber(head: string, pos: number): [string, number] {
	const end = numberEnd(head, pos);
	if (end < 0 || end - pos > MAX_NUMBER_CHARS) throw new HeadError();
	return [head.slice(pos, end), end];
}

function readString(head: string, pos: number): [string, number] {
	if (head.charAt(pos) !== '"') throw new HeadError();
	let p = pos + 1;
	let value = '';
	for (;;) {
		if (p >= head.length) throw new HeadError();
		const c = head.charAt(p);
		if (c === '"') {
			p++;
			break;
		}
		if (c === '\\') {
			const n = head.charAt(p + 1);
			if (n !== '"' && n !== '\\') throw new HeadError();
			value += n;
			p += 2;
			continue;
		}
		value += c;
		p++;
	}
	const length = Array.from(value).length;
	if (length < 1 || length > MAX_STRING_CHARS) throw new HeadError();
	return [value, p];
}

function readStringList(head: string, pos: number): [string[], number] {
	const values: string[] = [];
	let p = pos;
	for (;;) {
		const [value, next] = readString(head, p);
		p = next;
		values.push(value);
		if (values.length > MAX_VALUES) throw new HeadError();
		const probe = skipWs(head, p);
		if (probe < head.length && head.charAt(probe) === ',') {
			p = skipWs(head, probe + 1);
			continue;
		}
		return [values, p];
	}
}

function readVar(head: string, pos: number): [string, number] {
	if (head.charAt(pos) !== '$') throw new HeadError();
	const first = head.charAt(pos + 1);
	if (!isWord(first) || isDigit(first)) throw new HeadError();
	let p = pos + 2;
	while (isWord(head.charAt(p))) p++;
	if (p - pos - 1 > MAX_NAME_CHARS) throw new HeadError();
	return [head.slice(pos + 1, p), p];
}

function matchIsEmpty(head: string, pos: number): [EmptyOp, number] | null {
	if (!head.startsWith('is', pos)) return null;
	let p = skipWs(head, pos + 2);
	if (p === pos + 2) return null;
	let negated = false;
	if (head.startsWith('not', p)) {
		const q = skipWs(head, p + 3);
		if (q > p + 3 && head.startsWith('empty', q)) {
			negated = true;
			p = q;
		}
	}
	if (!head.startsWith('empty', p)) return null;
	const end = p + 5;
	if (isWord(head.charAt(end))) return null;
	return [negated ? 'is not empty' : 'is empty', end];
}

function matchOp(head: string, pos: number): [ConditionOp, number] | null {
	const word = (text: string): boolean =>
		head.startsWith(text, pos) && !isWord(head.charAt(pos + text.length));
	for (const sym of ['<=', '>=', '!=', '=', '<', '>'] as const) {
		if (head.startsWith(sym, pos)) {
			const end = pos + sym.length;
			return isWord(head.charAt(end)) ? null : [sym, end];
		}
	}
	if (word('in')) return ['in', pos + 2];
	if (word('contains')) return ['contains', pos + 8];
	if (head.startsWith('not', pos)) {
		const q = skipWs(head, pos + 3);
		if (q > pos + 3) {
			if (head.startsWith('in', q) && !isWord(head.charAt(q + 2))) return ['not in', q + 2];
			if (head.startsWith('contains', q) && !isWord(head.charAt(q + 8))) {
				return ['not contains', q + 8];
			}
		}
	}
	return null;
}

function parseTest(head: string, pos: number, base: number): [ConditionTest, number] {
	const start = pos;
	const [name, afterVar] = readVar(head, pos);
	let p = needWs(head, afterVar);
	const span = (end: number) => ({ var: name, start: base + start, end: base + end });

	const empty = matchIsEmpty(head, p);
	if (empty) return [{ ...span(empty[1]), kind: 'none', op: empty[0] as EmptyOp }, empty[1]];

	const matched = matchOp(head, p);
	if (!matched) throw new HeadError();
	const op = matched[0];
	p = needWs(head, matched[1]);

	if (op === '<' || op === '<=' || op === '>' || op === '>=') {
		const [number, end] = readNumber(head, p);
		return [{ ...span(end), kind: 'number', op, number }, end];
	}

	if (op === '=' || op === '!=') {
		if (head.charAt(p) === '"') {
			const [value, end] = readString(head, p);
			return [{ ...span(end), kind: 'string', op, strings: [value] }, end];
		}
		const [number, end] = readNumber(head, p);
		return [{ ...span(end), kind: 'number', op, number }, end];
	}

	if (op === 'contains' || op === 'not contains') {
		const [value, end] = readString(head, p);
		return [{ ...span(end), kind: 'string', op, strings: [value] }, end];
	}

	if (op !== 'in' && op !== 'not in') throw new HeadError();

	if (head.charAt(p) === '"') {
		const [values, end] = readStringList(head, p);
		return [{ ...span(end), kind: 'strings', op, strings: values }, end];
	}

	const [low, afterLow] = readNumber(head, p);
	if (!head.startsWith('..', afterLow)) throw new HeadError();
	const [high, end] = readNumber(head, afterLow + 2);
	if (cmpDec(toDec(low), toDec(high)) > 0) throw new HeadError();
	return [{ ...span(end), kind: 'range', op, low, high }, end];
}

export function parseCondition(head: string, base = 0): Condition | null {
	try {
		const tests: ConditionTest[] = [];
		let joiner: ConditionJoiner | null = null;
		let pos = 0;
		for (;;) {
			const [test, next] = parseTest(head, pos, base);
			pos = next;
			tests.push(test);
			if (tests.length > MAX_TESTS) return null;
			if (pos >= head.length) break;
			pos = needWs(head, pos);
			let word: ConditionJoiner | null = null;
			if (head.startsWith('and', pos) && !isWord(head.charAt(pos + 3))) word = 'and';
			else if (head.startsWith('or', pos) && !isWord(head.charAt(pos + 2))) word = 'or';
			if (word === null || (joiner !== null && word !== joiner)) return null;
			joiner = word;
			pos = needWs(head, pos + word.length);
		}
		return { joiner: joiner ?? 'and', tests, start: base, end: base + head.length };
	} catch (e) {
		if (e instanceof HeadError) return null;
		throw e;
	}
}

function formatString(value: string): string {
	let out = '"';
	for (const ch of value) {
		out += ch === '"' || ch === '\\' ? `\\${ch}` : ch;
	}
	return `${out}"`;
}

export function formatTest(test: ConditionTest): string {
	const head = `$${test.var} ${test.op}`;
	switch (test.kind) {
		case 'none':
			return head;
		case 'number':
			return `${head} ${test.number}`;
		case 'string':
			return `${head} ${formatString(test.strings[0] ?? '')}`;
		case 'strings':
			return `${head} ${test.strings.map(formatString).join(', ')}`;
		case 'range':
			return `${head} ${test.low}..${test.high}`;
	}
}

export function formatCondition(
	source: Condition | readonly ConditionTest[],
	joiner: ConditionJoiner = 'and'
): string {
	const tests = Array.isArray(source) ? source : (source as Condition).tests;
	const word = Array.isArray(source) ? joiner : (source as Condition).joiner;
	return tests.map(formatTest).join(` ${word} `);
}

export function sameCondition(a: Condition | null, b: Condition | null): boolean {
	if (a === null || b === null) return a === b;
	if (a.joiner !== b.joiner || a.tests.length !== b.tests.length) return false;
	return a.tests.every((x, i) => {
		const y = b.tests[i];
		if (!y) return false;
		const strip = (t: ConditionTest) =>
			JSON.stringify(t, (key, v) => (key === 'start' || key === 'end' ? undefined : v));
		return strip(x) === strip(y);
	});
}

function asNumber(value: string): string | null {
	const text = pyStrip(value);
	if (text.length > MAX_VALUE_NUMBER_CHARS) return null;
	return numberEnd(text, 0) === text.length ? text : null;
}

function has(values: ConditionValues, name: string): boolean {
	return Object.prototype.hasOwnProperty.call(values, name);
}

function evalTest(test: ConditionTest, values: ConditionValues): boolean {
	if (!has(values, test.var)) return false;
	const value = values[test.var];
	if (typeof value !== 'string') return false;

	switch (test.kind) {
		case 'none':
			return test.op === 'is empty' ? pyStrip(value) === '' : pyStrip(value) !== '';
		case 'number': {
			const text = asNumber(value);
			if (text === null) return false;
			const c = cmpDec(toDec(text), toDec(test.number));
			switch (test.op) {
				case '=':
					return c === 0;
				case '!=':
					return c !== 0;
				case '<':
					return c < 0;
				case '<=':
					return c <= 0;
				case '>':
					return c > 0;
				default:
					return c >= 0;
			}
		}
		case 'range': {
			const text = asNumber(value);
			if (text === null) return false;
			const n = toDec(text);
			const inside = cmpDec(toDec(test.low), n) <= 0 && cmpDec(n, toDec(test.high)) <= 0;
			return test.op === 'in' ? inside : !inside;
		}
		case 'string': {
			const needle = test.strings[0] ?? '';
			if (test.op === '=') return pyStrip(value) === needle;
			if (test.op === '!=') return pyStrip(value) !== needle;
			const found = casefold(value).includes(casefold(needle));
			return test.op === 'contains' ? found : !found;
		}
		case 'strings': {
			const hit = test.strings.includes(pyStrip(value));
			return test.op === 'in' ? hit : !hit;
		}
	}
}

export function evaluateCondition(cond: Condition, values: ConditionValues): boolean {
	if (cond.joiner === 'or') return cond.tests.some((t) => evalTest(t, values));
	return cond.tests.every((t) => evalTest(t, values));
}

export function chooseBranch(
	block: ConditionBlock,
	values: ConditionValues
): { position: number | null; body: string } {
	for (let position = 0; position < block.branches.length; position++) {
		const branch = block.branches[position];
		if (!branch) continue;
		if (branch.kind === 'else') return { position, body: branch.body };
		if (branch.cond !== null && evaluateCondition(branch.cond, values)) {
			return { position, body: branch.body };
		}
	}
	return { position: null, body: '' };
}

function matchOpen(text: string, i: number): { kind: 'if' | 'elseif'; end: number } | null {
	let p = i + 2;
	let kind: 'if' | 'elseif';
	if (text.startsWith('if', p) && isWs(text.charAt(p + 2))) {
		kind = 'if';
		p += 2;
	} else if (text.startsWith('else', p) && isWs(text.charAt(p + 4))) {
		p = skipWs(text, p + 4);
		if (!text.startsWith('if', p)) return null;
		p += 2;
		if (!isWs(text.charAt(p))) return null;
		kind = 'elseif';
	} else {
		return null;
	}
	return { kind, end: skipWs(text, p) };
}

function isHigh(code: number): boolean {
	return code >= 0xd800 && code <= 0xdbff;
}

function isLow(code: number): boolean {
	return code >= 0xdc00 && code <= 0xdfff;
}

function countCodePoints(text: string, from: number, to: number): number {
	let n = 0;
	for (let i = from; i < to; i++) {
		n++;
		if (isHigh(text.charCodeAt(i)) && i + 1 < to && isLow(text.charCodeAt(i + 1))) i++;
	}
	return n;
}

function advanceCodePoints(text: string, from: number, count: number): number {
	let i = from;
	for (let n = 0; n < count && i < text.length; n++) {
		i += isHigh(text.charCodeAt(i)) && isLow(text.charCodeAt(i + 1)) ? 2 : 1;
	}
	return i;
}

function readHeadEnd(text: string, start: number, limit: number): number {
	let p = start;
	while (p < limit) {
		const c = text.charAt(p);
		if (c === '\\') {
			if (p + 1 >= limit) break;
			p += 2;
		} else if (c === '"') {
			let q = p + 1;
			let closed = -1;
			while (q < limit) {
				const d = text.charAt(q);
				if (d === '\\') {
					if (q + 1 >= limit) break;
					q += 2;
				} else if (d === '"') {
					closed = q + 1;
					break;
				} else {
					q++;
				}
			}
			if (closed < 0) break;
			p = closed;
		} else if (c === '>') {
			if (p + 1 < limit && text.charAt(p + 1) === '>') break;
			p++;
		} else {
			p++;
		}
	}
	return p;
}

function matchJunk(text: string, i: number): number {
	const p = i + 2;
	for (const kw of ['if', 'else', 'end']) {
		if (!text.startsWith(kw, p)) continue;
		const after = text.charAt(p + kw.length);
		if (!isWs(after) && after !== '>') continue;
		const q = p + kw.length;
		let r = q;
		while (r - q < 60 && r < text.length) {
			const c = text.charAt(r);
			if (c === '>' || c === '\r' || c === '\n') break;
			r++;
		}
		return text.startsWith('>>', r) ? r + 2 : q;
	}
	return -1;
}

function readTag(text: string, i: number): Tok | null {
	if (text.startsWith('<<end>>', i)) {
		return { kind: 'end', start: i, end: i + 7, head: '', headStart: i + 7 };
	}
	if (text.startsWith('<<else>>', i)) {
		return { kind: 'else', start: i, end: i + 8, head: '', headStart: i + 8 };
	}
	const open = matchOpen(text, i);
	if (open) {
		const limit = advanceCodePoints(text, open.end, MAX_HEAD_CHARS + 2);
		const headEnd = readHeadEnd(text, open.end, limit);
		if (text.startsWith('>>', headEnd) && countCodePoints(text, open.end, headEnd) <= MAX_HEAD_CHARS) {
			const raw = text.slice(open.end, headEnd);
			const [lo, hi] = pyStripBounds(raw);
			return {
				kind: open.kind,
				start: i,
				end: headEnd + 2,
				head: raw.slice(lo, hi),
				headStart: open.end + lo
			};
		}
		return { kind: 'junk', start: i, end: open.end, head: '', headStart: open.end };
	}
	const junkEnd = matchJunk(text, i);
	if (junkEnd >= 0) return { kind: 'junk', start: i, end: junkEnd, head: '', headStart: junkEnd };
	return null;
}

function tokenize(text: string): [Tok[], number] {
	const toks: Tok[] = [];
	let pos = 0;
	let unitAt = 0;
	let cpAt = 0;
	for (;;) {
		const i = text.indexOf('<<', pos);
		if (i < 0) return [toks, pos];
		cpAt += countCodePoints(text, unitAt, i);
		unitAt = i;
		if (cpAt >= MAX_SCAN_CHARS) return [toks, pos];
		const tag = readTag(text, i);
		if (tag === null) {
			pos = i + 1;
		} else {
			toks.push(tag);
			pos = tag.end;
		}
	}
}

function scrub(input: string): string {
	let text = input;
	for (let pass = 0; pass < MAX_SCRUB_PASSES; pass++) {
		const [toks] = tokenize(text);
		if (toks.length === 0) break;
		const pieces: string[] = [];
		let cursor = 0;
		for (const tok of toks) {
			pieces.push(text.slice(cursor, tok.start));
			cursor = tok.end;
		}
		pieces.push(text.slice(cursor));
		text = pieces.join('');
	}
	return text;
}

function tailTagLength(text: string, i: number): number {
	if (text.startsWith('<<end>>', i)) return 7;
	if (text.startsWith('<<else>>', i)) return 8;
	for (const kw of ['if', 'else', 'end']) {
		if (!text.startsWith(kw, i + 2)) continue;
		const after = text.charAt(i + 2 + kw.length);
		if (isWs(after) || after === '>') return 2 + kw.length;
	}
	return 0;
}

function findTail(text: string, from: number): number {
	let pos = from;
	for (;;) {
		const i = text.indexOf('<<', pos);
		if (i < 0) return -1;
		if (tailTagLength(text, i) > 0) return i;
		pos = i + 1;
	}
}

function stripTail(text: string): string {
	const pieces: string[] = [];
	let cursor = 0;
	let pos = 0;
	for (;;) {
		const i = text.indexOf('<<', pos);
		if (i < 0) break;
		const length = tailTagLength(text, i);
		if (length > 0) {
			pieces.push(text.slice(cursor, i));
			cursor = i + length;
			pos = cursor;
		} else {
			pos = i + 1;
		}
	}
	pieces.push(text.slice(cursor));
	return pieces.join('');
}

function buildBlock(
	text: string,
	toks: readonly Tok[],
	lo: number,
	hi: number,
	index: number,
	errors: ConditionError[]
): ConditionBlock {
	const branches: ConditionBranch[] = [];
	let headTok: Tok | null = null;
	let parts: string[] = [];
	let cursor = 0;

	const flush = (bodyEnd: number): void => {
		if (headTok === null) return;
		const tok: Tok = headTok;
		let cond: Condition | null = null;
		if (tok.kind !== 'else') {
			cond = parseCondition(tok.head, tok.headStart);
			if (cond === null) errors.push({ kind: 'bad_head', offset: tok.start });
		}
		branches.push({
			kind: tok.kind as BranchKind,
			cond,
			head: tok.head,
			headStart: tok.headStart,
			tagStart: tok.start,
			tagEnd: tok.end,
			bodyStart: tok.end,
			bodyEnd,
			body: scrub(parts.join(''))
		});
	};

	for (let idx = lo; idx <= hi; idx++) {
		const tok = toks[idx];
		if (!tok) break;
		if (tok.kind === 'junk') {
			parts.push(text.slice(cursor, tok.start));
			cursor = tok.end;
			continue;
		}
		if (tok.kind === 'end') {
			parts.push(text.slice(cursor, tok.start));
			flush(tok.start);
			break;
		}
		if (headTok !== null) {
			parts.push(text.slice(cursor, tok.start));
			flush(tok.start);
		}
		headTok = tok;
		parts = [];
		cursor = tok.end;
	}
	return {
		index,
		start: toks[lo]?.start ?? 0,
		end: toks[hi]?.end ?? 0,
		branches
	};
}

interface OpenBlock {
	startIdx: number;
	hasElse: boolean;
	bad: boolean;
	branchCount: number;
}

export function parseTemplate(input: string): ParsedTemplate | null {
	if (!input.includes('<<')) return null;
	const text = input.replace(SENTINEL_STRIP, '');
	const [toks, tailStart] = tokenize(text);
	const tail = findTail(text, tailStart);
	if (toks.length === 0 && tail < 0) return null;

	const errors: ConditionError[] = [];
	const valid: Array<[number, number]> = [];
	let cur: OpenBlock | null = null;

	toks.forEach((tok, idx) => {
		const kind = tok.kind;
		if (kind === 'junk') {
			errors.push({ kind: 'malformed_tag', offset: tok.start });
			return;
		}
		if (kind === 'if') {
			if (cur !== null) errors.push({ kind: 'nested', offset: tok.start });
			cur = { startIdx: idx, hasElse: false, bad: false, branchCount: 1 };
			return;
		}
		if (cur === null) {
			errors.push({ kind: 'stray', offset: tok.start });
			return;
		}
		const open: OpenBlock = cur;
		if (kind === 'elseif') {
			if (open.hasElse) {
				open.bad = true;
				errors.push({ kind: 'stray', offset: tok.start });
			}
			open.branchCount++;
		} else if (kind === 'else') {
			if (open.hasElse) {
				open.bad = true;
				errors.push({ kind: 'stray', offset: tok.start });
			}
			open.hasElse = true;
			open.branchCount++;
		} else {
			if (open.branchCount > MAX_BRANCHES || valid.length >= MAX_BLOCKS) {
				open.bad = true;
				errors.push({ kind: 'limit', offset: toks[open.startIdx]?.start ?? 0 });
			}
			if (!open.bad) valid.push([open.startIdx, idx]);
			cur = null;
		}
	});
	const unclosed = cur as OpenBlock | null;
	if (unclosed !== null) {
		errors.push({ kind: 'unclosed', offset: toks[unclosed.startIdx]?.start ?? 0 });
	}

	const out: string[] = [];
	const blocks: ConditionBlock[] = [];
	let cursor = 0;
	let idx = 0;
	let vi = 0;
	while (idx < toks.length) {
		const tok = toks[idx] as Tok;
		const span = valid[vi];
		if (span && span[0] === idx) {
			const [lo, hi] = span;
			vi++;
			out.push(text.slice(cursor, tok.start));
			const blockIndex = blocks.length;
			blocks.push(buildBlock(text, toks, lo, hi, blockIndex, errors));
			out.push(`${SENTINEL_OPEN}${blockIndex}${SENTINEL_CLOSE}`);
			cursor = (toks[hi] as Tok).end;
			idx = hi + 1;
			continue;
		}
		out.push(text.slice(cursor, tok.start));
		cursor = tok.end;
		idx++;
	}
	out.push(tail < 0 ? text.slice(cursor) : stripTail(text.slice(cursor)));
	if (tail >= 0) errors.push({ kind: 'limit', offset: tail });
	errors.sort((a, b) => a.offset - b.offset);
	return { source: text, skeleton: scrub(out.join('')), blocks, errors };
}

function joinHole(leftIn: string, rightIn: string): string {
	let left = leftIn;
	let right = rightIn;
	const r0 = right.charAt(0);
	if (left.endsWith(',') && (right === '' || r0 === ',' || r0 === '\r' || r0 === '\n')) {
		if (right.startsWith(',')) right = trimBlankStart(right.slice(1));
		else left = trimBlankEnd(left.slice(0, -1));
	} else if (left === '' && right.startsWith(',')) {
		right = trimBlankStart(right.slice(1));
	}
	const l = left.charAt(left.length - 1);
	const r = right.charAt(0);
	if (!left || !right || l === '\r' || l === '\n' || r === '\r' || r === '\n' || r === ',') {
		return left + right;
	}
	return `${left} ${right}`;
}

function nextSentinel(text: string, from: number): { start: number; end: number; k: number } | null {
	let pos = from;
	for (;;) {
		const o = text.indexOf(SENTINEL_OPEN, pos);
		if (o < 0) return null;
		let p = o + 1;
		while (isDigit(text.charAt(p))) p++;
		if (p > o + 1 && text.charAt(p) === SENTINEL_CLOSE) {
			return { start: o, end: p + 1, k: Number(text.slice(o + 1, p)) };
		}
		pos = o + 1;
	}
}

export function splice(input: string, bodies: readonly string[]): string {
	const pieces: string[] = [];
	let cursor = 0;
	let m = nextSentinel(input, 0);
	while (m) {
		const body = bodies[m.k] ?? '';
		pieces.push(input.slice(cursor, m.start));
		pieces.push(hasContent(body) ? body.replace(SENTINEL_STRIP, '') : input.slice(m.start, m.end));
		cursor = m.end;
		m = nextSentinel(input, cursor);
	}
	pieces.push(input.slice(cursor));
	let text = pieces.join('');
	for (;;) {
		const hole = nextSentinel(text, 0);
		if (hole === null) return text;
		text = joinHole(trimBlankEnd(text.slice(0, hole.start)), trimBlankStart(text.slice(hole.end)));
	}
}

export function renderParsed(parsed: ParsedTemplate, values: ConditionValues): string {
	const bodies = parsed.blocks.map((block) => {
		const { body } = chooseBranch(block, values);
		return hasContent(body) ? body : '';
	});
	return splice(parsed.skeleton, bodies);
}

export function renderConditions(template: string, values: ConditionValues): RenderResult {
	const parsed = parseTemplate(template);
	if (parsed === null) return { text: template, errors: [] };
	return { text: renderParsed(parsed, values), errors: [...parsed.errors] };
}

export function conditionReads(parsed: ParsedTemplate): Set<string> {
	const names = new Set<string>();
	for (const block of parsed.blocks) {
		for (const branch of block.branches) {
			for (const test of branch.cond?.tests ?? []) names.add(test.var);
		}
	}
	return names;
}
