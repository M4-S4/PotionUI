export type InputModality = 'pointer' | 'keyboard';

const NAVIGATION_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown']);
const LISTEN = { capture: true, passive: true } as const;

export function installInputModality(doc: Document = document): () => void {
	const root = doc.documentElement;
	let heldNavigationKey: string | null = null;
	let enteringFromOutside = false;
	let focusedAtBlur: EventTarget | null = null;

	function setModality(modality: InputModality) {
		if (root.dataset.input !== modality) root.dataset.input = modality;
	}

	function releaseNavigationKey() {
		heldNavigationKey = null;
	}

	function onWindowBlur() {
		enteringFromOutside = true;
		focusedAtBlur = doc.activeElement;
	}

	function onPointerDown() {
		releaseNavigationKey();
		enteringFromOutside = false;
		setModality('pointer');
	}

	function onKeyDown(event: KeyboardEvent) {
		if (event.ctrlKey || event.metaKey || event.altKey) return;
		if (event.key === 'Tab') setModality('keyboard');
		else if (NAVIGATION_KEYS.has(event.key)) heldNavigationKey = event.key;
	}

	function onKeyUp(event: KeyboardEvent) {
		if (event.key === heldNavigationKey) releaseNavigationKey();
	}

	function onFocusIn(event: FocusEvent) {
		const enteredElsewhere = enteringFromOutside && event.target !== focusedAtBlur;
		enteringFromOutside = false;
		if (heldNavigationKey !== null || enteredElsewhere) setModality('keyboard');
	}

	const view = doc.defaultView;
	doc.addEventListener('pointerdown', onPointerDown, LISTEN);
	doc.addEventListener('keydown', onKeyDown, LISTEN);
	doc.addEventListener('keyup', onKeyUp, LISTEN);
	doc.addEventListener('focusin', onFocusIn, LISTEN);
	view?.addEventListener('blur', onWindowBlur);

	return () => {
		doc.removeEventListener('pointerdown', onPointerDown, LISTEN);
		doc.removeEventListener('keydown', onKeyDown, LISTEN);
		doc.removeEventListener('keyup', onKeyUp, LISTEN);
		doc.removeEventListener('focusin', onFocusIn, LISTEN);
		view?.removeEventListener('blur', onWindowBlur);
		delete root.dataset.input;
	};
}
