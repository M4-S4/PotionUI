import { describe, expect, it } from 'vitest';
import { annotateLockHints } from './lockHints';
import { processSchemaWithReactions } from './reactions';

const schema = () => ({
	properties: {
		root: {
			children: [
				{ name: 'speed_profile', type: 'select', label: 'Speed profile' },
				{
					name: 'steps',
					type: 'slider',
					label: 'Steps (0 = checkpoint default)',
					description: 'How many sampling steps.',
					reactions: [
						{ when: { field: 'speed_profile', equals: 'auto' }, then: { set_value: 0, set_disabled: true } },
						{ when: { field: 'speed_profile', equals: 'custom' }, then: { set_disabled: false } }
					]
				},
				{ name: 'seed', type: 'seed' }
			]
		}
	}
});

function run(profile: string) {
	const { processedSchema } = processSchemaWithReactions(schema(), { speed_profile: profile });
	annotateLockHints(processedSchema, { speed_profile: profile });
	return processedSchema.properties.root.children as Record<string, any>[];
}

describe('annotateLockHints', () => {
	it('tells the user which control sets a locked field', () => {
		const steps = run('auto')[1];
		expect(steps.disabled).toBe(true);
		expect(steps.description).toBe('How many sampling steps. Set by Speed profile: auto.');
	});

	it('leaves unlocked fields alone', () => {
		const children = run('custom');
		expect(children[1].description).toBe('How many sampling steps.');
		expect(children[2].description).toBeUndefined();
	});

	it('adds a hint even when the field had no description', () => {
		const { processedSchema } = processSchemaWithReactions(
			{
				properties: {
					root: {
						children: [
							{ name: 'speed_profile', label: 'Speed profile' },
							{
								name: 'cfg',
								label: 'CFG',
								reactions: [{ when: { field: 'speed_profile', equals: 'auto' }, then: { set_disabled: true } }]
							}
						]
					}
				}
			},
			{ speed_profile: 'auto' }
		);
		annotateLockHints(processedSchema, { speed_profile: 'auto' });
		expect(processedSchema.properties.root.children[1].description).toBe('Set by Speed profile: auto.');
	});
});
