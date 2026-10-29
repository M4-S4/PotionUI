import { describe, it, expect, vi, beforeEach } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { writable } from 'svelte/store';
import { mountComponent } from './helpers/mount';

const { default: EditorSaveControls } = await import('$lib/media/editors/EditorSaveControls.svelte');
const { EDITOR_RUN_CONTEXT_KEY } = await import('$lib/media/editors/editorRunContext');

let cleanups: Array<() => void> = [];

function render(props: Record<string, unknown>, run?: { progress: number | null; cancel: () => void }) {
	const progress = writable<number | null>(run?.progress ?? null);
	const cancelling = writable(false);
	const target = document.createElement('div');
	document.body.appendChild(target);
	const context = run ? new Map([[EDITOR_RUN_CONTEXT_KEY, { progress, cancelling, cancel: run.cancel }]]) : undefined;
	const instance = mount(EditorSaveControls as never, {
		target,
		context,
		props: { onCancel: vi.fn(), onSave: vi.fn(), ...props }
	});
	flushSync();
	cleanups.push(() => {
		unmount(instance);
		target.remove();
	});
	return { target, progress, cancelling };
}

const labels = (root: ParentNode) => Array.from(root.querySelectorAll('button')).map((b) => (b.textContent || '').trim());

beforeEach(() => {
	cleanups.forEach((fn) => fn());
	cleanups = [];
	document.body.innerHTML = '';
});

describe('EditorSaveControls', () => {
	it('offers Save as new and Replace original with Cancel and Apply', () => {
		const { target } = render({});
		expect(labels(target)).toEqual(['Save as new', 'Replace original', 'Cancel', 'Apply']);
	});

	it('sends new until the other choice is made, then that one', () => {
		const onSave = vi.fn();
		const { target } = render({ onSave });
		const apply = () => (Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Apply') as HTMLButtonElement).click();

		apply();
		expect(onSave).toHaveBeenLastCalledWith('new');
		(Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Replace original') as HTMLButtonElement).click();
		flushSync();
		apply();
		expect(onSave).toHaveBeenLastCalledWith('replace');
	});

	it('hides the choice and always sends new when replacing is not allowed', () => {
		const onSave = vi.fn();
		const { target } = render({ allowReplace: false, onSave, note: 'Saved as a new image.' });

		expect(labels(target)).toEqual(['Cancel', 'Apply']);
		expect(target.querySelector('[data-tool-note]')?.textContent).toBe('Saved as a new image.');
		(target.querySelectorAll('button')[1] as HTMLButtonElement).click();
		expect(onSave).toHaveBeenCalledWith('new');
	});

	it('draws the check only on the plain Apply, and uses the label it is given otherwise', () => {
		const plain = render({});
		expect(plain.target.querySelector('.ml-auto button:last-child svg')).not.toBeNull();
		const frame = render({ allowReplace: false, applyLabel: 'Save frame' });
		expect(labels(frame.target)).toEqual(['Cancel', 'Save frame']);
		expect(frame.target.querySelector('.ml-auto button:last-child svg')).toBeNull();
	});

	it('disables Apply with the reason, and shows a failure', () => {
		const { target } = render({ blockedReason: 'Nothing to apply yet.' });
		expect((Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Apply') as HTMLButtonElement).disabled).toBe(true);
		expect(target.querySelector('[data-tool-blocked]')?.textContent).toBe('Nothing to apply yet.');

		const failed = render({ failureMessage: 'The edit failed' });
		expect(failed.target.querySelector('[data-tool-failure]')?.textContent).toBe('The edit failed');
	});

	it('says Working with a disabled Cancel while it saves something that is not a run', () => {
		const { target } = render({ busy: true });
		expect(target.querySelector('[data-tool-working]')?.textContent).toContain('Working…');
		expect(target.querySelector('[data-tool-progress]')).toBeNull();
		expect(labels(target)).toEqual(['Cancel']);
		expect((target.querySelector('button') as HTMLButtonElement).disabled).toBe(true);
	});

	it('follows the run the editor started and cancels it from the footer', () => {
		const cancel = vi.fn();
		const { target, progress } = render({ busy: true }, { progress: 0.4, cancel });

		expect(target.querySelector('[data-tool-progress]')?.textContent).toContain('40%');
		expect(labels(target)).toEqual(['Cancel edit']);

		progress.set(0.9);
		flushSync();
		expect(target.querySelector('[data-tool-progress]')?.textContent).toContain('90%');

		(target.querySelector('button') as HTMLButtonElement).click();
		flushSync();
		expect(cancel).toHaveBeenCalledTimes(1);
	});

	it('shows Cancelling while the run says it is being cancelled', () => {
		const { target, cancelling } = render({ busy: true }, { progress: 0.4, cancel: vi.fn() });
		cancelling.set(true);
		flushSync();
		expect(labels(target)).toEqual(['Cancelling']);
		expect((target.querySelector('button') as HTMLButtonElement).disabled).toBe(true);
	});

	it('asks for the run to be cancelled only while there is one', () => {
		const cancel = vi.fn();
		const { target } = render({ busy: true }, { progress: null, cancel });
		expect(target.querySelector('[data-tool-working]')).not.toBeNull();
		expect(labels(target)).toEqual(['Cancel']);
		expect(cancel).not.toHaveBeenCalled();
	});

	it('lets the user press Cancel edit again when the server refused the first cancel', () => {
		const cancel = vi.fn();
		const { target, cancelling } = render({ busy: true }, { progress: 0.4, cancel });
		cancelling.set(true);
		flushSync();
		cancelling.set(false);
		flushSync();

		expect(labels(target)).toEqual(['Cancel edit']);
		(target.querySelector('button') as HTMLButtonElement).click();
		expect(cancel).toHaveBeenCalledTimes(1);
	});

	it('sends new, not a replace picked earlier, once the choice is no longer offered', () => {
		const onSave = vi.fn();
		const view = mountComponent(EditorSaveControls as never, { onCancel: vi.fn(), onSave, allowReplace: true });
		const press = (text: string) =>
			(Array.from(view.target.querySelectorAll('button')).find((b) => b.textContent?.trim() === text) as HTMLButtonElement).click();

		press('Replace original');
		flushSync();
		view.rerender({ allowReplace: false });
		press('Apply');

		expect(onSave).toHaveBeenCalledTimes(1);
		expect(onSave).toHaveBeenCalledWith('new');
	});

	it('does not bring the old pick back when the choice is offered again', () => {
		const onSave = vi.fn();
		const view = mountComponent(EditorSaveControls as never, { onCancel: vi.fn(), onSave, allowReplace: true });
		const press = (text: string) =>
			(Array.from(view.target.querySelectorAll('button')).find((b) => b.textContent?.trim() === text) as HTMLButtonElement).click();

		press('Replace original');
		flushSync();
		view.rerender({ allowReplace: false });
		view.rerender({ allowReplace: true });
		const pressed = view.target.querySelector('[role="group"] [aria-pressed="true"]');
		expect(pressed?.textContent?.trim()).toBe('Save as new');
		press('Apply');
		expect(onSave).toHaveBeenCalledWith('new');
	});
});
