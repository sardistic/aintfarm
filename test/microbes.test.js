import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEvolvedWorld } from '../public/evolution.js';
import { createMicrobialWorld } from '../public/microbes.js';

function randomFrom(seedValue) {
  let seed = seedValue;
  return () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
}

test('gathering grows a visible organism and eventually divides it', () => {
  const world = makeEvolvedWorld(1);
  let absorbed = 0;
  const culture = { consumeAt: () => { absorbed += 2; return 2; }, releaseAt: () => {} };
  const microbes = createMicrobialWorld(world, culture, 1, randomFrom(17));
  const founder = microbes.organisms[0], firstRadius = founder.radius;
  const chemicalBefore = world.hormones.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < 500; i++) microbes.step(.05);
  const report = microbes.summary();
  assert.ok(absorbed > 20);
  assert.ok(report.captures > 20);
  assert.ok(report.largest > firstRadius);
  assert.ok(report.divisions > 0);
  assert.ok(microbes.ancestry.some(node => node.parentId === founder.id));
  assert.ok(microbes.snapshots.length >= 4);
  assert.ok(world.hormones.reduce((sum, value) => sum + value, 0) > chemicalBefore);
});

test('predation damages a weaker family and returns a dead body to the culture', () => {
  const world = makeEvolvedWorld(1);
  let released = 0;
  const culture = { consumeAt: () => 0, releaseAt: (_x, _y, amount) => { released += amount; } };
  const microbes = createMicrobialWorld(world, culture, 2, randomFrom(29));
  const [predator, prey] = microbes.organisms;
  predator.x = prey.x = 500; predator.y = prey.y = 320;
  predator.mass = 120; predator.radius = 3 + Math.sqrt(predator.mass) * .72;
  predator.genome.aggression = 1; predator.attackClock = 0;
  prey.mass = 25; prey.radius = 3 + Math.sqrt(prey.mass) * .72;
  prey.genome.aggression = 0; prey.energy = 2;
  microbes.step(.05);
  assert.ok(microbes.summary().attacks >= 1);
  assert.ok(microbes.summary().deaths >= 1);
  assert.ok(released > 0);
  assert.ok(predator.mass > 120);
  assert.ok(microbes.events.some(event => event.type === 'kill' && event.target === prey.id));
});
