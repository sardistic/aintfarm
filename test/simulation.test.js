import test from 'node:test';
import assert from 'node:assert/strict';
import { makeWorld, stepWorld, cellAt, capacity, needsDigging, editSand, nextStep, applyJev, GROUND, ENTRANCE } from '../public/simulation.js';
import { createBrain, evaluateBrain, rewardBrain } from '../public/brain.js';

test('black sand holds a starter shaft and workers move grains for brood space', () => {
  const world = makeWorld(28), initialCapacity = capacity(world);
  assert.equal(cellAt(world, 3, GROUND + 5), 1);
  assert.equal(cellAt(world, ENTRANCE, GROUND + 5), 0);
  assert.equal(needsDigging(world), true);
  for (let i = 0; i < 600; i++) stepWorld(world, .05);
  assert.ok(world.excavated > 0);
  assert.ok(world.spoiled > 0);
  assert.ok(world.spoiled <= world.excavated);
  assert.ok(capacity(world) > initialCapacity);
});

test('a forager can discover food and carry it to the brood chamber', () => {
  const world = makeWorld(1), ant = world.ants[0], food = world.foods[0];
  ant.role = 'forager'; ant.brainClock = 0; ant.clock = 1;
  ant.x = food.x; ant.y = GROUND - 1;
  stepWorld(world, .05);
  assert.equal(ant.cargo, 'food');
  ant.x = ENTRANCE; ant.y = 33; ant.clock = 1;
  stepWorld(world, .05);
  assert.equal(world.delivered, 1);
  assert.equal(ant.cargo, null);
});

test('sand can be moved, removed, added, and loose sand falls', () => {
  const world = makeWorld(0);
  const source = { x: 100, y: 210 }, destination = { x: 100, y: 80 };
  assert.equal(editSand(world, 'move', source, destination), true);
  assert.equal(cellAt(world, 10, 21), 0);
  assert.equal(cellAt(world, 10, 8), 2);
  for (let i = 0; i < 5; i++) stepWorld(world, .05);
  assert.equal(cellAt(world, 10, 8), 0);
  assert.equal(editSand(world, 'remove', { x: 110, y: 210 }), true);
  assert.equal(cellAt(world, 11, 21), 0);
  assert.equal(editSand(world, 'add', { x: 110, y: 210 }), true);
  assert.equal(cellAt(world, 11, 21), 2);
  assert.equal(editSand(world, 'food', { x: 160, y: 70 }), true);
  assert.equal(world.foods.length, 5);
});

test('ants route through open cells and Jev changes excavation pressure', () => {
  const world = makeWorld(1);
  const step = nextStep(world, { x: ENTRANCE, y: 33 }, { x: ENTRANCE, y: GROUND - 1 });
  assert.deepEqual(step, { x: ENTRANCE, y: 32 });
  const previousPressure = world.pressure;
  applyJev(world, { chambers: .1, branches: .2, balance: .5 });
  assert.ok(world.pressure > previousPressure);
  assert.ok(world.branching > 1);
});

test('an individual recurrent brain changes task when its sensed need changes', () => {
  const brain = createBrain('digger');
  let task = 'digger';
  for (let i = 0; i < 12; i++) task = evaluateBrain(brain, [1, 0, 0, 0, 0, 0, 0, 1], task);
  assert.equal(task, 'digger');
  for (let i = 0; i < 20; i++) task = evaluateBrain(brain, [0, 1, 0, 0, 0, 1, 0, 0], task);
  assert.equal(task, 'forager');
  assert.equal(brain.inputWeights.length, 64);
  assert.equal(brain.outputWeights.length, 32);
  const before = brain.bias[1];
  rewardBrain(brain, 'forager');
  assert.ok(brain.bias[1] > before);
});

test('a large colony runs food, brood care, births, and dynamic tasks', () => {
  const world = makeWorld(400);
  for (let i = 0; i < 600; i++) stepWorld(world, .05);
  assert.ok(world.ants.length >= 400);
  assert.ok(world.births > 0);
  assert.ok(world.delivered > 0);
  assert.ok(world.ants.some(ant => ant.brain.ticks > 0));
  assert.ok(new Set(world.ants.map(ant => ant.role)).size >= 3);
});
