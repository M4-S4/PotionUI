import { test, expect } from '@playwright/test';
import { SIGNED_OUT } from './helpers';

test.use({ storageState: SIGNED_OUT });

const chipMarkup = (n: number) =>
	`<span class="chip phrase-chip"><button type="button" class="chip-main"><span class="chip-mark">#</span><span class="chip-context">Lighting</span><span class="chip-label">Golden hour backlight ${n}</span></button></span>`;

test('phrasebook chips on consecutive wrapped lines do not overlap', async ({ page }) => {
	await page.goto('/');

	await page.evaluate((markup) => {
		const host = document.createElement('div');
		host.id = 'chip-spacing-host';
		host.className = 'composer';
		host.style.cssText = 'width: 320px; position: fixed; top: 40px; left: 40px;';
		host.innerHTML = `<div class="segment-content"><div class="inline-chip-editor" style="white-space: pre-wrap; word-break: break-word;">${markup}</div></div>`;
		document.body.appendChild(host);
	}, [1, 2, 3, 4].map(chipMarkup).join(' '));

	const boxes = await page.$$eval('#chip-spacing-host .chip', (els) =>
		els.map((el) => {
			const r = el.getBoundingClientRect();
			return { top: r.top, bottom: r.bottom, left: r.left, right: r.right };
		})
	);
	expect(boxes.length).toBe(4);

	const tops = new Set(boxes.map((b) => Math.round(b.top)));
	expect(tops.size).toBeGreaterThan(1);

	for (let i = 0; i < boxes.length; i++) {
		for (let j = i + 1; j < boxes.length; j++) {
			const a = boxes[i];
			const b = boxes[j];
			const intersects = a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
			expect(intersects, `chip ${i} and ${j} intersect`).toBe(false);
		}
	}

	const lineHeight = await page.$eval('#chip-spacing-host .inline-chip-editor', (el) =>
		parseFloat(getComputedStyle(el).lineHeight)
	);
	for (const b of boxes) expect(b.bottom - b.top).toBeLessThanOrEqual(lineHeight);
});

const choiceChipMarkup = (label: string) =>
	`<span class="chip choice-chip choice-group-chip" contenteditable="false" data-chip="choice"><button type="button" class="chip-main"><span class="chip-mark">{ }</span><span class="chip-label">${label}</span></button><button type="button" class="chip-config" aria-label="Edit choice group"></button></span>`;

const conditionChipMarkup =
	'<span class="chip condition-chip" data-chip="condition"><button type="button" class="chip-main"><span class="chip-mark">if</span><span class="chip-context">age</span><span class="chip-label">older</span></button></span>';

const editorMarkup = (inner: string) => `<div class="inline-chip-editor" style="white-space: pre-wrap; word-break: break-word;">${inner}</div>`;

const chipMarkSurfaces: Record<string, string> = {
	generate: `<div class="segment-composer"><div class="composer"><div class="card segment"><div class="card-content segment-content">${editorMarkup(
		`${choiceChipMarkup('glowing · shimmering')} ${chipMarkup(1)} ${conditionChipMarkup}`
	)}</div></div></div></div>`,
	library: `<div class="segment-composer"><div class="segment-content">${editorMarkup(
		`${choiceChipMarkup('glowing · shimmering')} ${chipMarkup(1)} ${conditionChipMarkup}`
	)}</div></div>`
};

for (const width of [1440, 390]) {
	for (const theme of ['dark', 'light']) {
		test(`choice group chip mark stays inside its box and clear of its label at ${width}px in ${theme}`, async ({ page }) => {
			await page.setViewportSize({ width, height: 800 });
			await page.goto(`/login?theme=${theme}`);
			await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

			await page.evaluate(
				(surfaces) => {
					const host = document.createElement('div');
					host.id = 'chip-mark-host';
					host.style.cssText = 'position: fixed; top: 16px; left: 16px; right: 16px; display: grid; gap: 24px;';
					host.innerHTML = Object.entries(surfaces)
						.map(([name, markup]) => `<div data-surface="${name}">${markup}</div>`)
						.join('');
					document.body.appendChild(host);
				},
				chipMarkSurfaces
			);
			await page.evaluate(async () => {
				await document.fonts.load('600 12px "IBM Plex Mono"');
				await document.fonts.ready;
			});

			const measured = await page.evaluate(() => {
				const box = (el: Element | Range) => {
					const r = el.getBoundingClientRect();
					return { left: r.left, right: r.right, width: r.width };
				};
				return [...document.querySelectorAll<HTMLElement>('#chip-mark-host [data-surface]')].map((surface) => {
					const chip = surface.querySelector('.choice-chip')!;
					const mark = chip.querySelector<HTMLElement>('.chip-mark')!;
					const label = chip.querySelector('.chip-label')!;
					const glyphs = document.createRange();
					glyphs.selectNodeContents(mark);
					return {
						surface: surface.dataset.surface,
						chip: box(chip),
						mark: box(mark),
						glyphs: box(glyphs),
						label: box(label),
						markOverflow: mark.scrollWidth - mark.clientWidth,
						phraseMarkWidth: surface.querySelector('.phrase-chip .chip-mark')!.getBoundingClientRect().width,
						conditionMarkWidth: surface.querySelector('.condition-chip .chip-mark')!.getBoundingClientRect().width
					};
				});
			});
			expect(measured.map((m) => m.surface)).toEqual(['generate', 'library']);

			for (const m of measured) {
				expect(m.markOverflow, `${m.surface} mark text overflows its box`).toBeLessThanOrEqual(0);
				expect(m.glyphs.left, `${m.surface} braces start outside the mark`).toBeGreaterThanOrEqual(m.mark.left);
				expect(m.glyphs.right, `${m.surface} braces end outside the mark`).toBeLessThanOrEqual(m.mark.right);
				expect(m.mark.left, `${m.surface} mark leaves the chip`).toBeGreaterThanOrEqual(m.chip.left);
				expect(m.mark.right, `${m.surface} mark box runs into the label`).toBeLessThanOrEqual(m.label.left);
				expect(m.glyphs.right, `${m.surface} braces run into the label`).toBeLessThanOrEqual(m.label.left);
				expect(m.label.left - m.mark.right, `${m.surface} gap before the label`).toBeGreaterThanOrEqual(4);
				expect(m.phraseMarkWidth, `${m.surface} phrasebook mark width`).toBe(16);
				expect(m.conditionMarkWidth, `${m.surface} condition mark width`).toBe(16);
			}
		});
	}
}
