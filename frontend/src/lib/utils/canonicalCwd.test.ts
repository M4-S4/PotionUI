import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { rerunFromCanonicalCwd, type CwdHost } from '../../../canonicalCwd';

const LOWERCASE = 'g:\\pinokio\\api\\potionui-pinokio.git\\app\\frontend';
const CANONICAL = 'G:\\pinokio\\api\\potionui-pinokio.git\\app\\frontend';
const VITE_LOWERCASE = `${LOWERCASE}\\node_modules\\vite\\bin\\vite.js`;
const VITE_CANONICAL = `${CANONICAL}\\node_modules\\vite\\bin\\vite.js`;

function host(overrides: Partial<CwdHost> = {}) {
	const calls = {
		cwd: vi.fn(() => LOWERCASE),
		realpath: vi.fn((path: string) => path.replace(LOWERCASE, CANONICAL)),
		rerun: vi.fn(() => 3),
		exit: vi.fn()
	};
	const value: CwdHost = {
		platform: 'win32',
		env: { PATH: 'C:\\node' },
		execPath: 'C:\\node\\node.exe',
		execArgv: ['--max-old-space-size=4096'],
		argv: ['C:\\node\\node.exe', VITE_LOWERCASE, 'build', '--mode', 'production'],
		...calls,
		...overrides
	};
	return { value, calls };
}

describe('rerunFromCanonicalCwd', () => {
	it('reruns the same command from the on-disk casing of a lowercase-drive folder and exits with its code', () => {
		const { value, calls } = host();
		rerunFromCanonicalCwd(value);
		expect(calls.rerun).toHaveBeenCalledTimes(1);
		expect(calls.rerun).toHaveBeenCalledWith(
			'C:\\node\\node.exe',
			['--max-old-space-size=4096', VITE_CANONICAL, 'build', '--mode', 'production'],
			CANONICAL,
			{ PATH: 'C:\\node', POTIONUI_CANONICAL_CWD: '1' }
		);
		expect(calls.exit).toHaveBeenCalledWith(3);
	});

	it.each(['linux', 'darwin'])('does nothing on %s', (platform) => {
		const { value, calls } = host({ platform });
		rerunFromCanonicalCwd(value);
		expect(calls.cwd).not.toHaveBeenCalled();
		expect(calls.rerun).not.toHaveBeenCalled();
		expect(calls.exit).not.toHaveBeenCalled();
	});

	it('leaves an already canonical working directory alone', () => {
		const { value, calls } = host({ cwd: vi.fn(() => CANONICAL) });
		rerunFromCanonicalCwd(value);
		expect(calls.rerun).not.toHaveBeenCalled();
		expect(calls.exit).not.toHaveBeenCalled();
	});

	it('never reruns twice, even if the child still sees a lowercase drive', () => {
		const { value, calls } = host({ env: { POTIONUI_CANONICAL_CWD: '1' } });
		rerunFromCanonicalCwd(value);
		expect(calls.rerun).not.toHaveBeenCalled();
	});

	it('never relocates the project to a different directory', () => {
		const { value, calls } = host({ realpath: vi.fn(() => 'D:\\elsewhere\\frontend') });
		rerunFromCanonicalCwd(value);
		expect(calls.rerun).not.toHaveBeenCalled();
	});

	it('keeps running in place when the folder cannot be resolved', () => {
		const { value, calls } = host({
			realpath: vi.fn(() => {
				throw new Error('EPERM');
			})
		});
		expect(() => rerunFromCanonicalCwd(value)).not.toThrow();
		expect(calls.rerun).not.toHaveBeenCalled();
	});

	it('keeps the script path as given when only the script cannot be resolved', () => {
		const { value, calls } = host({
			realpath: vi.fn((path: string) => {
				if (path === LOWERCASE) return CANONICAL;
				throw new Error('ENOENT');
			})
		});
		rerunFromCanonicalCwd(value);
		expect(calls.rerun).toHaveBeenCalledWith(
			'C:\\node\\node.exe',
			['--max-old-space-size=4096', VITE_LOWERCASE, 'build', '--mode', 'production'],
			CANONICAL,
			expect.any(Object)
		);
	});
});

describe('vite.config.ts', () => {
	const source = readFileSync(new URL('../../../vite.config.ts', import.meta.url), 'utf-8');

	it('does not import SvelteKit statically, which would load it from the lowercase path before the rerun', () => {
		expect(source).not.toMatch(/^import[^;]*from\s+['"]@sveltejs\/kit\/vite['"]/m);
	});

	it('reruns from the canonical folder before SvelteKit is loaded', () => {
		const rerun = source.indexOf('rerunFromCanonicalCwd();');
		const loaded = source.indexOf("await import('@sveltejs/kit/vite')");
		expect(rerun).toBeGreaterThan(-1);
		expect(loaded).toBeGreaterThan(rerun);
	});
});
