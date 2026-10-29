import { describe, it, expect } from 'vitest';
import {
	editsTheResource,
	hasEditor,
	isSurfaceEditorKind,
	isToolDialogKind,
	MEDIA_EDITOR_KINDS,
	toolDialogKind,
	toolIdOfKind,
	type MediaEditorKind
} from './types';

const KINDS: readonly MediaEditorKind[] = MEDIA_EDITOR_KINDS;

describe('hasEditor', () => {
	it('offers the paint editor on images only', () => {
		expect(hasEditor('paint', 'image')).toBe(true);
		expect(hasEditor('paint', 'video')).toBe(false);
		expect(hasEditor('paint', 'audio')).toBe(false);
	});

	it('offers crop and mask on images only', () => {
		expect(hasEditor('crop', 'image')).toBe(true);
		expect(hasEditor('mask', 'image')).toBe(true);
		expect(hasEditor('crop', 'video')).toBe(false);
		expect(hasEditor('crop', 'audio')).toBe(false);
		expect(hasEditor('mask', 'video')).toBe(false);
		expect(hasEditor('mask', 'audio')).toBe(false);
	});

	it('offers a frame grab on video only — a still has no frames to lift', () => {
		expect(hasEditor('frame', 'video')).toBe(true);
		expect(hasEditor('frame', 'image')).toBe(false);
		expect(hasEditor('frame', 'audio')).toBe(false);
	});

	it('offers trim on both timed media', () => {
		expect(hasEditor('trim', 'video')).toBe(true);
		expect(hasEditor('trim', 'audio')).toBe(true);
		expect(hasEditor('trim', 'image')).toBe(false);
	});

	it('offers split on audio only', () => {
		expect(hasEditor('split', 'audio')).toBe(true);
		expect(hasEditor('split', 'video')).toBe(false);
		expect(hasEditor('split', 'image')).toBe(false);
	});

	it('offers resize on images and video, never on audio', () => {
		expect(hasEditor('resize', 'image')).toBe(true);
		expect(hasEditor('resize', 'video')).toBe(true);
		expect(hasEditor('resize', 'audio')).toBe(false);
	});

	it('offers nothing when the kind is unknown', () => {
		for (const kind of KINDS) {
			expect(hasEditor(kind, null)).toBe(false);
		}
	});
});

describe('editsTheResource', () => {
	it('is true for the four that change the media', () => {
		expect(editsTheResource('crop')).toBe(true);
		expect(editsTheResource('trim')).toBe(true);
		expect(editsTheResource('frame')).toBe(true);
		expect(editsTheResource('split')).toBe(true);
		expect(editsTheResource('resize')).toBe(true);
	});

	it('is false for a mask, which is stored beside the media and changes no row', () => {
		expect(editsTheResource('mask')).toBe(false);
	});

	it('is false for the paint editor, which saves a new upload and needs no row', () => {
		expect(editsTheResource('paint')).toBe(false);
	});
});

describe('tool dialog kinds', () => {
	it('names a catalog tool without an editor by its id', () => {
		expect(toolDialogKind('rotate')).toBe('tool:rotate');
		expect(toolDialogKind('acme.loop')).toBe('tool:acme.loop');
	});

	it('tells a dialog kind from an editor kind and gets the id back', () => {
		expect(isToolDialogKind('tool:rotate')).toBe(true);
		expect(isToolDialogKind('crop')).toBe(false);
		expect(toolIdOfKind('tool:acme.loop')).toBe('acme.loop');
	});

	it('offers a dialog on any known media kind and none on an unknown one', () => {
		for (const media of ['image', 'video', 'audio'] as const) {
			expect(hasEditor('tool:rotate', media)).toBe(true);
		}
		expect(hasEditor('tool:rotate', null)).toBe(false);
	});

	it('edits the resource, so it needs one behind it', () => {
		expect(editsTheResource('tool:rotate')).toBe(true);
	});
});

describe('isSurfaceEditorKind', () => {
	it('knows the built-in surfaces and not the painter, the mask editor or a made-up name', () => {
		for (const name of ['crop', 'trim', 'frame', 'split', 'resize']) {
			expect(isSurfaceEditorKind(name)).toBe(true);
		}
		for (const name of ['paint', 'mask', 'nonsense', '', null, undefined]) {
			expect(isSurfaceEditorKind(name)).toBe(false);
		}
	});
});
