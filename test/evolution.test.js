import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEvolvedWorld, stepEvolvedWorld, performAction, actionMotivation, editSand, paintSand, cellAt, GROUND } from '../public/evolution.js';
import { ACTIONS, PARAMETERS, createGenome, createController, chooseAction } from '../public/evolved-brain.js';
import { CELL, COLS, addGrain, stepGrains } from '../public/granular.js';
import { emitSignals } from '../public/signals.js';
import { chemicalResponse } from '../public/signals.js';

test('offspring inherit a mutated 9,904-weight controller and a lineage', () => {
  const world = makeEvolvedWorld(1), parent = world.ants[0];
  assert.equal(PARAMETERS, 9904);
  parent.energy = 110; parent.harvests = 2;
  performAction(world, parent, ACTIONS.indexOf('reproduce'));
  assert.equal(world.births, 1);
  const child = world.ants[1];
  assert.equal(child.parentId, parent.id);
  assert.equal(child.lineage, parent.lineage);
  assert.equal(child.generation, 1);
  assert.equal(child.brain.genome.length, PARAMETERS);
  assert.notStrictEqual(child.brain.genome, parent.brain.genome);
  assert.ok(child.brain.genome.some((weight, i) => weight !== parent.brain.genome[i]));
  assert.ok(child.mutation.neuralWeights > 0);
  assert.ok(child.mutation.changes.length > 0);
  assert.notStrictEqual(child.receptor, parent.receptor);
  assert.ok(parent.energy < 110);
  assert.deepEqual(world.events.at(-1).type, 'birth');
  assert.equal(world.events.at(-1).parent, parent.id);
  assert.equal(world.familyTree.at(-1).parentId, parent.id);
  assert.equal(world.familyTree.at(-1).founderFamily, parent.lineage);
  assert.equal(world.familyTree.at(-1).mutation.neuralWeights, child.mutation.neuralWeights);
  assert.equal(world.attempts[ACTIONS.indexOf('reproduce')], 1);
  assert.equal(world.actions[ACTIONS.indexOf('reproduce')], 1);
});

test('failed reproduction is an attempt, not a birth', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.energy = 110;
  assert.equal(performAction(world, ant, ACTIONS.indexOf('reproduce')), false);
  assert.equal(world.attempts[ACTIONS.indexOf('reproduce')], 1);
  assert.equal(world.actions[ACTIONS.indexOf('reproduce')], 0);
});

test('stored food reaches the nest and colony signals bias movement', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 250; ant.y = 165; ant.bodyX = 501; ant.bodyY = 331;
  ant.foodCargo = 1; ant.energy = 90; ant.clock = -100;
  stepEvolvedWorld(world, .05);
  assert.equal(world.foodDelivered, 1);
  assert.equal(world.foodStored, 1);
  ant.foodCargo = 0; ant.energy = 120;
  ant.signature.fill(0); ant.signature[0] = 1;
  ant.receptor.fill(0); ant.receptor[0] = 1;
  emitSignals(world, 503, 331, ant.signature, 5);
  const weights = actionMotivation(world, ant);
  assert.ok(weights[1] > weights[3]);
});

test('surface preference keeps a forager out of the shaft', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 250; ant.y = GROUND - 1;
  ant.bodyX = 501; ant.bodyY = (GROUND - .5) * CELL;
  ant.traits.foraging = 1;
  assert.equal(performAction(world, ant, ACTIONS.indexOf('south')), false);
  ant.traits.foraging = 0;
  assert.equal(performAction(world, ant, ACTIONS.indexOf('south')), true);
});

test('matching chemical mutations cause a cooperative encounter', () => {
  const world = makeEvolvedWorld(2), [a, b] = world.ants;
  for (const ant of [a, b]) {
    ant.x = 250; ant.y = 165; ant.bodyX = 501; ant.bodyY = 331;
    ant.signature.fill(1); ant.receptor.fill(1);
    ant.traits.sociability = .9; ant.clock = -100;
  }
  a.energy = 80; b.energy = 20;
  assert.ok(chemicalResponse(a, b) > 0);
  for (let i = 0; i < 11; i++) stepEvolvedWorld(world, .05);
  assert.ok(world.cooperations > 0);
  assert.ok(b.energy > 20);
});

test('opposed receptors and aggressive traits cause conflict', () => {
  const world = makeEvolvedWorld(2), [a, b] = world.ants;
  for (const ant of [a, b]) {
    ant.x = 250; ant.y = 165; ant.bodyX = 501; ant.bodyY = 331;
    ant.clock = -100; ant.energy = 80;
  }
  a.receptor.fill(-1); b.signature.fill(1); a.traits.aggression = .95;
  b.traits.aggression = .1;
  for (let i = 0; i < 11; i++) stepEvolvedWorld(world, .05);
  assert.ok(world.conflicts > 0);
  assert.ok(b.energy < a.energy);
});

test('nest food storage has a finite capacity', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 250; ant.y = 165; ant.bodyX = 501; ant.bodyY = 331;
  ant.energy = 90; ant.foodCargo = 2; ant.clock = -100;
  world.foodStored = 239;
  stepEvolvedWorld(world, .05);
  assert.equal(world.foodStored, 240);
  assert.equal(world.foodDelivered, 2);
  assert.equal(world.foodWasted, 1);
});

test('physical actions expose buried food and move black sand', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 283; ant.y = 165; ant.energy = 90;
  assert.equal(cellAt(world, 284, 165), 1);
  performAction(world, ant, ACTIONS.indexOf('dig east'));
  assert.equal(cellAt(world, 284, 165), 0);
  assert.equal(ant.cargo, 'sand');
  const grainId = ant.cargoGrain;
  assert.equal(world.particles[grainId].kind, 3);
  performAction(world, ant, ACTIONS.indexOf('drop sand'));
  assert.equal(ant.cargo, null);
  assert.equal(world.particles[grainId].kind, 2);
  assert.equal(world.spoiled, 1);
  ant.x = 50; ant.y = GROUND - 1;
  const before = ant.energy;
  performAction(world, ant, ACTIONS.indexOf('harvest'));
  assert.ok(ant.energy > before);
  assert.equal(world.delivered, 1);
  assert.deepEqual(world.events.map(event => event.type), ['dig', 'drop', 'harvest']);
});

test('a buried food patch can only be harvested after its grain is removed', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 283; ant.y = 165; ant.energy = 50;
  const food = world.foods.find(source => source.x === 284 && source.y === 165);
  const before = food.amount;
  performAction(world, ant, ACTIONS.indexOf('harvest'));
  assert.equal(food.amount, before);
  performAction(world, ant, ACTIONS.indexOf('dig east'));
  assert.equal(cellAt(world, 284, 165), 0);
  performAction(world, ant, ACTIONS.indexOf('harvest'));
  assert.equal(food.amount, before - 1);
});

test('an agent accelerates smoothly through open space', () => {
  const world = makeEvolvedWorld(1), ant = world.ants[0];
  ant.x = 250; ant.y = 165;
  ant.bodyX = 501; ant.bodyY = 331; ant.clock = -100;
  const start = ant.bodyX;
  performAction(world, ant, ACTIONS.indexOf('east'));
  stepEvolvedWorld(world, .05);
  assert.ok(ant.bodyX > start && ant.bodyX < start + CELL);
  assert.ok(ant.vx > 0);
});

test('agents act from neural outputs and the population changes over time', () => {
  const world = makeEvolvedWorld(60);
  for (let i = 0; i < 600; i++) stepEvolvedWorld(world, .05);
  assert.ok(world.ants.some(ant => ant.brain.ticks > 0));
  assert.ok(world.births > 0);
  assert.ok(world.excavated > 0);
  assert.ok(world.delivered > 0);
  assert.ok(world.actions.reduce((sum, count) => sum + count, 0) > 100);
  assert.ok(world.snapshots.length >= 3);
  assert.equal(world.snapshots[0].time, 0);
});

test('nutrients are autonomous rather than placed by the user', () => {
  const world = makeEvolvedWorld(0);
  const count = world.foods.length;
  assert.ok(count > 0);
  assert.equal(editSand(world, 'food', { x: 340, y: 80 }), false);
  assert.equal(world.foods.length, count);
});

test('a loose grain accelerates under gravity and stops at the packed surface', () => {
  const world = makeEvolvedWorld(0);
  const grain = addGrain(world, 30, 8);
  assert.ok(grain);
  const start = grain.y;
  for (let i = 0; i < 100; i++) {
    world.time += .05;
    stepGrains(world, .05);
  }
  assert.ok(grain.y > start + CELL * 10);
  assert.ok(grain.y < GROUND * CELL);
  assert.equal(cellAt(world, Math.floor(grain.x / CELL), Math.floor(grain.y / CELL)), 2);
  assert.equal(world.activeGrains.has(grain.id), false);
});

test('dragging transfers one persistent grain into the physics simulation', () => {
  const world = makeEvolvedWorld(0);
  const grainId = world.grainAt[GROUND * COLS + 12];
  const source = { x: 12 * CELL + CELL / 2, y: GROUND * CELL + CELL / 2 };
  const destination = { x: 12 * CELL + CELL / 2, y: 8 * CELL + CELL / 2 };
  assert.equal(editSand(world, 'move', source, destination), true);
  assert.equal(world.particles[grainId].kind, 2);
  assert.equal(world.particles[grainId].x, destination.x);
  assert.equal(world.particles[grainId].y, destination.y);
  assert.equal(cellAt(world, 12, GROUND), 0);
  assert.equal(world.activeGrains.has(grainId), true);
  const initialY = world.particles[grainId].y;
  for (let i = 0; i < 12; i++) { world.time += .05; stepGrains(world, .05); }
  assert.ok(world.particles[grainId].y > initialY);
});

test('pouring along a fast pointer stroke fills the whole line', () => {
  const world = makeEvolvedWorld(0);
  assert.equal(paintSand(world, 'add', { x: 100, y: 40 }, { x: 300, y: 40 }), true);
  assert.ok(world.looseGrains.size >= 300);
  const centers = new Set([...world.looseGrains].map(id => world.particles[id])
    .filter(grain => grain.y === 40).map(grain => Math.round(grain.x)));
  for (let x = 100; x <= 300; x += CELL) assert.ok(centers.has(x), `missing grain at ${x}`);
  const before = world.looseGrains.size;
  assert.equal(paintSand(world, 'remove', { x: 100, y: 40 }, { x: 300, y: 40 }), true);
  assert.ok(world.looseGrains.size < before / 2);
});

test('a dense stream stays outside cohesive packed sand', () => {
  const world = makeEvolvedWorld(0);
  paintSand(world, 'add', { x: 100, y: 60 }, { x: 160, y: 60 });
  for (let i = 0; i < 120; i++) { world.time += .05; stepGrains(world, .05); }
  for (const id of world.looseGrains) {
    const grain = world.particles[id];
    const x = Math.floor(grain.x / CELL), y = Math.floor(grain.y / CELL);
    assert.notEqual(world.sand[y * COLS + x], 1);
  }
});

test('energy depletion removes agents without usable food', () => {
  const world = makeEvolvedWorld(4);
  world.foods = [];
  for (const ant of world.ants) ant.energy = .01;
  for (let i = 0; i < 10; i++) stepEvolvedWorld(world, .05);
  assert.equal(world.ants.length, 0);
  assert.equal(world.deaths, 4);
});

test('controllers use stochastic actions without changing weights during thought', () => {
  const genome = createGenome(), brain = createController(genome);
  const before = genome.slice();
  const action = chooseAction(brain, new Float32Array(32), () => .5);
  assert.ok(action >= 0 && action < ACTIONS.length);
  assert.deepEqual(genome, before);
  assert.equal(brain.ticks, 1);
});
