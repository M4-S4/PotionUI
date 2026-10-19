import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SparkRenderer, SplatFileType, SplatMesh } from '@sparkjsdev/spark';
import type { SplatContainerFormat } from '../splatUrl';
import {
	DEFAULT_FOV_Y,
	frameFromCenters,
	sampleStride,
	viewFromFraming,
	viewFromSource,
	type CameraView
} from './splatFraming';
import { parseSourceView } from './splatHeader';
import { SPLAT_BACKGROUND, SplatLoadError, splatErrorFromStatus } from './splatSupport';

const CENTER_SAMPLE_LIMIT = 200_000;
const AUTO_ROTATE_SPEED = 2;

const FILE_TYPES: Record<SplatContainerFormat, SplatFileType> = {
	ply: SplatFileType.PLY,
	spz: SplatFileType.SPZ,
	splat: SplatFileType.SPLAT
};

export interface SplatViewer {
	load(
		url: string,
		format: SplatContainerFormat | null,
		signal: AbortSignal
	): Promise<{ splatCount: number }>;
	clear(): void;
	resetView(): void;
	setAutoRotate(enabled: boolean): void;
	dispose(): void;
}

export interface SplatViewerOptions {
	container: HTMLElement;
	onContextLost?: () => void;
}

function abortError(): DOMException {
	return new DOMException('Aborted', 'AbortError');
}

export function createSplatViewer({ container, onContextLost }: SplatViewerOptions): SplatViewer {
	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
	} catch (error) {
		throw new SplatLoadError('webgl', null, error);
	}
	renderer.setClearColor(SPLAT_BACKGROUND, 1);
	renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

	const canvas = renderer.domElement;
	canvas.style.display = 'block';
	canvas.style.width = '100%';
	canvas.style.height = '100%';
	container.appendChild(canvas);

	let frameId = 0;
	let disposed = false;
	let splat: SplatMesh | null = null;
	let initialView: CameraView | null = null;

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(DEFAULT_FOV_Y, 1, 0.01, 1000);
	const group = new THREE.Group();
	group.rotation.x = Math.PI;
	scene.add(group);

	const spark = new SparkRenderer({ renderer, enableLod: false, onDirty: requestRender });
	scene.add(spark);

	const controls = new OrbitControls(camera, canvas);
	controls.enableDamping = true;
	controls.dampingFactor = 0.08;
	controls.screenSpacePanning = true;
	controls.autoRotateSpeed = AUTO_ROTATE_SPEED;
	controls.addEventListener('change', requestRender);

	function requestRender() {
		if (disposed || frameId) return;
		frameId = requestAnimationFrame(renderFrame);
	}

	function renderFrame() {
		frameId = 0;
		if (disposed) return;
		const moved = controls.update();
		renderer.render(scene, camera);
		if (moved || controls.autoRotate) requestRender();
	}

	function resize() {
		const width = Math.max(1, container.clientWidth);
		const height = Math.max(1, container.clientHeight);
		renderer.setSize(width, height, false);
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		requestRender();
	}

	const resizeObserver = new ResizeObserver(resize);
	resizeObserver.observe(container);
	resize();

	canvas.addEventListener('webglcontextlost', (event) => {
		event.preventDefault();
		onContextLost?.();
	});

	function centersOf(mesh: SplatMesh, total: number): Float32Array {
		const stride = sampleStride(total, CENTER_SAMPLE_LIMIT);
		const centers = new Float32Array(Math.ceil(total / stride) * 3);
		let written = 0;
		mesh.forEachSplat((index, center) => {
			if (index % stride !== 0) return;
			centers[written++] = center.x;
			centers[written++] = center.y;
			centers[written++] = center.z;
		});
		return centers.subarray(0, written);
	}

	function applyView(view: CameraView) {
		camera.fov = view.fov;
		camera.near = view.near;
		camera.far = view.far;
		camera.position.set(...view.position);
		camera.updateProjectionMatrix();

		controls.target.set(...view.target);
		controls.minDistance = view.minDistance;
		controls.maxDistance = view.maxDistance;
		controls.update();
	}

	function contentView(mesh: SplatMesh, total: number): CameraView | null {
		const framing = frameFromCenters(centersOf(mesh, total));
		if (!framing) return null;
		group.updateMatrixWorld(true);
		const center = new THREE.Vector3(...framing.center);
		group.localToWorld(center);
		return viewFromFraming([center.x, center.y, center.z], framing.radius, camera.aspect);
	}

	function clear() {
		if (splat) {
			group.remove(splat);
			splat.dispose();
			splat = null;
		}
		initialView = null;
		requestRender();
	}

	async function load(url: string, format: SplatContainerFormat | null, signal: AbortSignal) {
		clear();

		let bytes: Uint8Array;
		try {
			const response = await fetch(url, { signal });
			if (!response.ok) throw splatErrorFromStatus(response.status);
			bytes = new Uint8Array(await response.arrayBuffer());
		} catch (error) {
			if (signal.aborted) throw abortError();
			if (error instanceof SplatLoadError) throw error;
			throw new SplatLoadError('network', null, error);
		}
		if (signal.aborted || disposed) throw abortError();

		const sourceView = parseSourceView(bytes);
		const mesh = new SplatMesh({
			fileBytes: bytes,
			fileType: format ? FILE_TYPES[format] : undefined
		});
		try {
			await mesh.initialized;
		} catch (error) {
			mesh.dispose();
			throw new SplatLoadError('decode', null, error);
		}
		if (signal.aborted || disposed) {
			mesh.dispose();
			throw abortError();
		}

		const total = mesh.numSplats;
		const view = total > 0 ? (sourceView ? viewFromSource(sourceView) : contentView(mesh, total)) : null;
		if (!view) {
			mesh.dispose();
			throw new SplatLoadError('empty');
		}

		group.add(mesh);
		splat = mesh;
		initialView = view;
		applyView(view);
		requestRender();
		return { splatCount: total };
	}

	function resetView() {
		if (!initialView) return;
		applyView(initialView);
		requestRender();
	}

	function setAutoRotate(enabled: boolean) {
		controls.autoRotate = enabled;
		requestRender();
	}

	function dispose() {
		if (disposed) return;
		disposed = true;
		if (frameId) cancelAnimationFrame(frameId);
		resizeObserver.disconnect();
		controls.dispose();
		if (splat) {
			group.remove(splat);
			splat.dispose();
			splat = null;
		}
		scene.remove(spark);
		spark.dispose();
		renderer.dispose();
		renderer.forceContextLoss();
		canvas.remove();
	}

	return { load, clear, resetView, setAutoRotate, dispose };
}
