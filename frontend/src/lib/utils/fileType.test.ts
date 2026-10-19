import { describe, it, expect } from 'vitest';
import {
	normalizeFileType,
	isFileType,
	isImageFileType,
	isVideoFileType,
	isAudioFileType,
	isMeshFileType,
	isSplatFileType
} from './fileType';

describe('normalizeFileType', () => {
	it('lowercases a DB-row-shaped UPPERCASE value', () => {
		expect(normalizeFileType('MESH')).toBe('mesh');
	});

	it('trims surrounding whitespace', () => {
		expect(normalizeFileType('  Video  ')).toBe('video');
	});

	it('returns an empty string for non-string input', () => {
		expect(normalizeFileType(undefined)).toBe('');
		expect(normalizeFileType(null)).toBe('');
		expect(normalizeFileType(42)).toBe('');
		expect(normalizeFileType({})).toBe('');
	});
});

describe('isMeshFileType', () => {
	it('matches the WebSocket-shaped lowercase value', () => {
		expect(isMeshFileType('mesh')).toBe(true);
	});

	it('matches the history/DB-row-shaped UPPERCASE value', () => {
		expect(isMeshFileType('MESH')).toBe(true);
	});

	it('matches mixed casing', () => {
		expect(isMeshFileType('Mesh')).toBe(true);
	});

	it('does not match a different file type', () => {
		expect(isMeshFileType('image')).toBe(false);
		expect(isMeshFileType('IMAGE')).toBe(false);
	});

	it('does not match undefined or non-string values', () => {
		expect(isMeshFileType(undefined)).toBe(false);
		expect(isMeshFileType(null)).toBe(false);
		expect(isMeshFileType(123)).toBe(false);
	});
});

describe('isSplatFileType', () => {
	it('matches the WebSocket-shaped lowercase value and the history/DB-row-shaped UPPERCASE value', () => {
		expect(isSplatFileType('splat')).toBe(true);
		expect(isSplatFileType('SPLAT')).toBe(true);
		expect(isSplatFileType(' Splat ')).toBe(true);
	});

	it('is distinct from a mesh', () => {
		expect(isSplatFileType('mesh')).toBe(false);
		expect(isMeshFileType('splat')).toBe(false);
		expect(isMeshFileType('SPLAT')).toBe(false);
	});

	it('does not match undefined or non-string values', () => {
		expect(isSplatFileType(undefined)).toBe(false);
		expect(isSplatFileType(null)).toBe(false);
		expect(isSplatFileType(123)).toBe(false);
	});
});

describe('isImageFileType / isVideoFileType / isAudioFileType', () => {
	it('are case-insensitive the same way isMeshFileType is', () => {
		expect(isImageFileType('IMAGE')).toBe(true);
		expect(isVideoFileType('Video')).toBe(true);
		expect(isAudioFileType('AUDIO')).toBe(true);
	});
});

describe('isFileType', () => {
	it('is the shared primitive every isXFileType helper is built on', () => {
		expect(isFileType('MESH', 'mesh')).toBe(true);
		expect(isFileType('mesh', 'image')).toBe(false);
		expect(isFileType('SPLAT', 'splat')).toBe(true);
	});
});
