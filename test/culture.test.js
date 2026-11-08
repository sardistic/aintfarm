import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEvolvedWorld } from '../public/evolution.js';
import { createCulture } from '../public/culture.js';

test('motile cells travel, feed, and turn over with inherited mutations', () => {
  const world = makeEvolvedWorld(30);
  let seed = 271828;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const culture = createCulture(world, 320, random);
  const starting = new Float32Array(320 * 2);
  for (let i = 0; i < 320; i++) {
    starting[i * 2] = culture.vertices[i * 12 + 6];
    starting[i * 2 + 1] = culture.vertices[i * 12 + 7];
  }
  const initialFood = world.foods.reduce((sum, food) => sum + food.amount, 0);
  for (let i = 0; i < 400; i++) culture.step(.05);
  let traveled = 0;
  for (let i = 0; i < 320; i++) {
    const px = culture.vertices[i * 12 + 6], py = culture.vertices[i * 12 + 7];
    assert.ok(Number.isFinite(px) && Number.isFinite(py));
    assert.ok(px >= 2 && px <= 998 && py >= 2 && py <= 648);
    traveled += Math.hypot(px - starting[i * 2], py - starting[i * 2 + 1]);
  }
  assert.ok(traveled / 320 > 100);
  assert.ok(culture.births > 0);
  assert.ok(culture.macroMutations > 0);
  assert.ok(world.foods.reduce((sum, food) => sum + food.amount, 0) < initialFood);
  assert.equal(culture.summary().channels.reduce((sum, count) => sum + count, 0), 320);
  const firstX = culture.vertices[6], firstY = culture.vertices[7];
  assert.equal(culture.consumeAt(firstX, firstY, 1, 1), 1);
  assert.equal(culture.summary().harvested, 1);
});
