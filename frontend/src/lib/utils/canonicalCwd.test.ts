import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { enterCanonicalCwd, type CwdHost } from '../../../canonicalCwd';

const LOWERCASE = 'g:\\pinokio\\api\\potionui-pinokio.git\\app\\frontend';
const CANONICAL = 'G:\\pinokio\\api\\potionui-pinokio.git\\app\\frontend';

function host(overrides: Partial<CwdHost> = {}) {
	const calls = {
		cwd: vi.fn(() => LOWERCASE),
		chdir: vi.fn(),
		realpath: vi.fn(() => CANONICAL)
	};
	const value: CwdHost = { platform: 'win32', ...calls, ...overrides };
	return { value, calls };
}

describe('enterCanonicalCwd', () => {
	it('moves a lowercase-drive working directory onto the on-disk casing on Windows', () => {
		const { value, calls } = host();
		enterCanonicalCwd(value);
		expect(calls.realpath).toHaveBeenCalledWith(LOWERCASE);
		expect(calls.chdir).toHaveBeenCalledTimes(1);
		expect(calls.chdir).toHaveBeenCalledWith(CANONICAL);
	});

	it.each(['linux', 'darwin'])('does nothing on %s', (platform) => {
		const { value, calls } = host({ platform });
		enterCanonicalCwd(value);
		expect(calls.cwd).not.toHaveBeenCalled();
		expect(calls.realpath).not.toHaveBeenCalled();
		expect(calls.chdir).not.toHaveBeenCalled();
	});

	it('leaves an already canonical working directory alone', () => {
		const { value, calls } = host({ cwd: vi.fn(() => CANONICAL) });
		enterCanonicalCwd(value);
		expect(calls.chdir).not.toHaveBeenCalled();
	});

	it('never relocates the project to a different directory', () => {
		const { value, calls } = host({ realpath: vi.fn(() => 'D:\\elsewhere\\frontend') });
		enterCanonicalCwd(value);
		expect(calls.chdir).not.toHaveBeenCalled();
	});

	it('keeps the current directory when the path cannot be resolved', () => {
		const { value, calls } = host({
			realpath: vi.fn(() => {
				throw new Error('EPERM');
			})
		});
		expect(() => enterCanonicalCwd(value)).not.toThrow();
		expect(calls.chdir).not.toHaveBeenCalled();
	});
});

describe('vite.config.ts', () => {
	const source = readFileSync(new URL('../../../vite.config.ts', import.meta.url), 'utf-8');

	it('does not import SvelteKit statically, which would capture the working directory before it is fixed', () => {
		expect(source).not.toMatch(/^import[^;]*from\s+['"]@sveltejs\/kit\/vite['"]/m);
	});

	it('fixes the working directory before SvelteKit is loaded', () => {
		const fixed = source.indexOf('enterCanonicalCwd();');
		const loaded = source.indexOf("await import('@sveltejs/kit/vite')");
		expect(fixed).toBeGreaterThan(-1);
		expect(loaded).toBeGreaterThan(fixed);
	});
});
