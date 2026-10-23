import { test } from 'node:test';
import assert from 'node:assert/strict';
import { accentOf, toneOf, usageLabelOf } from './tone.js';

test('usage below 50 is success, 50 to 79 is warning, 80 and above is danger', () => {
    assert.equal(toneOf(0), 'success');
    assert.equal(toneOf(49.9), 'success');
    assert.equal(toneOf(50), 'warning');
    assert.equal(toneOf(79.9), 'warning');
    assert.equal(toneOf(80), 'danger');
    assert.equal(toneOf(100), 'danger');
});

test('the usage label follows the same thresholds', () => {
    assert.equal(usageLabelOf(49), 'Plenty of headroom');
    assert.equal(usageLabelOf(50), 'Moderate load');
    assert.equal(usageLabelOf(79), 'Moderate load');
    assert.equal(usageLabelOf(80), 'Running near capacity');
});

test('an accent is a semantic token reference, never a literal colour', () => {
    assert.equal(accentOf('success'), 'rgb(var(--success))');
    assert.match(accentOf(toneOf(10)), /^rgb\(var\(--[a-z]+\)\)$/);
});
