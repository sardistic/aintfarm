import test from 'node:test';
import assert from 'node:assert/strict';
import { clampCamera, panCamera, zoomCamera } from '../public/camera.js';

test('zoom keeps the point under the cursor fixed and clamps the visible area', () => {
  const camera = zoomCamera({ zoom: 1, x: 0, y: 0 }, 3, 240, 180, 800, 600);
  assert.deepEqual(camera, { zoom: 3, x: -480, y: -360 });
  assert.equal((240 - camera.x) / camera.zoom, 240);
  assert.equal((180 - camera.y) / camera.zoom, 180);
  assert.deepEqual(zoomCamera(camera, 1, 240, 180, 800, 600), { zoom: 1, x: 0, y: 0 });
});

test('panning cannot expose space outside the farm', () => {
  const camera = { zoom: 2, x: -250, y: -200 };
  assert.deepEqual(panCamera(camera, 1000, 1000, 800, 600), { zoom: 2, x: 0, y: 0 });
  assert.deepEqual(panCamera(camera, -1000, -1000, 800, 600), { zoom: 2, x: -800, y: -600 });
  assert.deepEqual(clampCamera({ zoom: 2, x: -700, y: -500 }, 400, 300), { zoom: 2, x: -400, y: -300 });
});
