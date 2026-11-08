import { WIDTH, HEIGHT, CELL, COLS, ROWS, GROUND, ENTRANCE, HOME_Y, AIR, PACKED, LOOSE,
  makeTerrain, cellAt, takeGrain, dropGrain, addGrain, pourGrain, removeGrain, stepGrains } from './granular.js';
import { ACTIONS, createGenome, createController, chooseAction } from './evolved-brain.js';
import { makeSignals, stepSignals, emitSignals, signalPreference, chemicalResponse } from './signals.js';
import { makePhenotype } from './traits.js';

export { WIDTH, HEIGHT, CELL, COLS, ROWS, GROUND, ENTRANCE, cellAt };
const directions = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const index = (x, y) => y * COLS + x;
const inside = (x, y) => x >= 0 && x < COLS && y >= 0 && y < ROWS;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const rand = (min, max) => min + Math.random() * (max - min);
const near = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y) <= 2;
const open = (world, x, y) => inside(x, y) && y >= GROUND - 1 && cellAt(world, x, y) !== PACKED;
const canEnter = (world, ant, x, y) => open(world, x, y)
  && !(ant.traits.foraging >= .5 && ant.y < GROUND && y >= GROUND && ant.cargo !== 'sand');
const MAX_EVENTS = 1500;
const MAX_STORED_FOOD = 240;

export function recordEvent(world, type, ant, details = {}) {
  const event = { time: Math.round(world.time * 100) / 100, type,
    agent: ant?.id ?? null, lineage: ant?.lineage ?? null,
    generation: ant?.generation ?? null, x: ant?.x ?? null, y: ant?.y ?? null, ...details };
  world.events.push(event);
  if (world.events.length > MAX_EVENTS) world.events.shift();
  world.eventRevision++;
  return event;
}

function recordSnapshot(world) {
  const generations = world.ants.map(ant => ant.generation);
  world.snapshots.push({ time: Math.round(world.time * 100) / 100,
    population: world.ants.length, lineages: new Set(world.ants.map(ant => ant.lineage)).size,
    deepestGeneration: generations.length ? Math.max(...generations) : 0,
    births: world.births, deaths: world.deaths, harvests: world.delivered,
    foodDelivered: world.foodDelivered, foodStored: world.foodStored,
    foodConsumed: world.foodConsumed, foodWasted: world.foodWasted,
    grainsDug: world.excavated, chemicalEncounters: world.chemicalEncounters,
    cooperations: world.cooperations, conflicts: world.conflicts, blooms: world.blooms,
    mutationBursts: world.mutationBursts,
    actions: Array.from(world.actions), attempts: Array.from(world.attempts) });
}

function makeCreature(world, x, y, parent = null) {
  const id = world.nextId++;
  const phenotype = makePhenotype(parent);
  const mutation = { ...phenotype.mutation, rate: .012 + (parent?.traits.mutability ?? .4) * .07 };
  const genome = createGenome(parent?.brain.genome, parent ? mutation : null);
  const ant = { id, parentId: parent?.id ?? null, lineage: parent?.lineage ?? id,
    generation: parent ? parent.generation + 1 : 0, x, y,
    bodyX: (x + .5) * CELL, bodyY: (y + .5) * CELL,
    prevBodyX: (x + .5) * CELL, prevBodyY: (y + .5) * CELL, vx: 0, vy: 0,
    energy: parent ? 36 : rand(58, 82), age: parent ? 0 : rand(0, 120),
    cargo: null, cargoAge: 0, foodCargo: 0, harvests: 0,
    lastBirth: -Infinity, signature: phenotype.signature, receptor: phenotype.receptor,
    traits: phenotype.traits, mutation: parent ? mutation : null,
    brain: createController(genome), clock: rand(0, .34),
    action: 14, lastAction: 14, color: phenotype.hue,
    encounterCooldown: 0 };
  world.familyTree?.push({ id, parentId: ant.parentId, founderFamily: ant.lineage,
    generation: ant.generation, bornAt: +world.time.toFixed(2), color: ant.color,
    traits: { ...ant.traits }, emission: [...ant.signature], reception: [...ant.receptor],
    mutation: ant.mutation });
  return ant;
}

export function makeEvolvedWorld(population = 180) {
  const world = makeTerrain();
  world.ants = [];
  world.foods = [50, 95, 145, 195, 305, 355, 405, 450].map(x =>
    ({ x, y: GROUND - 1, amount: 46, max: 65, regen: .42 }));
  world.foods.push(...[[242, GROUND - 1], [258, GROUND - 1]]
    .map(([x, y]) => ({ x, y, amount: 40, max: 55, regen: .45 })));
  world.foods.push(...[[244, 165], [256, 165]]
    .map(([x, y]) => ({ x, y, amount: 45, max: 60, regen: .6 })));
  // Buried patches make excavation useful: opening a packed cell exposes food.
  world.foods.push(...[
    [215, 160], [284, 165], [250, 198], [190, 213], [310, 215],
    [150, 153], [350, 153], [118, 238], [382, 238], [70, 280], [430, 280]
  ].map(([x, y]) => ({ x, y, amount: 65, max: 85, regen: .24 })));
  world.foodField = new Float32Array(world.sand.length);
  world.homeField = new Float32Array(world.sand.length);
  makeSignals(world);
  world.occupancy = new Uint16Array(world.sand.length);
  world.nextId = 1;
  // A resource-linked ceiling prevents an early baby boom from consuming the
  // entire food budget before later generations can establish themselves.
  world.populationLimit = Math.min(900, Math.max(population + 50, Math.round(population * 2.3)));
  world.regenClock = 0;
  world.fieldClock = 0;
  world.homeRevision = -1;
  world.actions = new Uint32Array(ACTIONS.length);
  world.attempts = new Uint32Array(ACTIONS.length);
  world.brood = 0;
  world.foodStored = 0;
  world.foodDelivered = 0;
  world.foodConsumed = 0;
  world.foodWasted = 0;
  world.delivered = 0;
  world.spoiled = 0;
  world.excavated = 0;
  world.births = 0;
  world.deaths = 0;
  world.chemicalEncounters = 0;
  world.cooperations = 0;
  world.conflicts = 0;
  world.blooms = 0;
  world.mutationBursts = 0;
  world.events = [];
  world.familyTree = [];
  world.eventRevision = 0;
  world.snapshots = [];
  world.snapshotClock = 0;
  for (let i = 0; i < population; i++) {
    const surface = i % 3 !== 0;
    const x = surface ? Math.floor(rand(25, COLS - 25)) : Math.floor(rand(235, 268));
    const y = surface ? GROUND - 1 : Math.floor(rand(155, 175));
    if (!open(world, x, y)) { i--; continue; }
    world.ants.push(makeCreature(world, x, y));
  }
  updateFields(world);
  stepSignals(world);
  recordSnapshot(world);
  return world;
}

function refreshFoodSources(world) {
  world.exposedFood = [];
  world.buriedFood = [];
  for (const food of world.foods) {
    if (food.amount < 1) continue;
    (open(world, food.x, food.y) ? world.exposedFood : world.buriedFood).push(food);
  }
}

function updateFields(world) {
  // The home field is a physical signal that follows accessible cells, not a route for any ant.
  if (world.homeRevision !== world.packedRevision) {
    const field = world.homeField;
    field.fill(0);
    const queue = new Int32Array(field.length);
    let head = 0, tail = 0;
    const source = index(ENTRANCE, HOME_Y);
    field[source] = 1;
    queue[tail++] = source;
    while (head < tail) {
      const current = queue[head++], x = current % COLS, y = Math.floor(current / COLS);
      const next = field[current] * .94;
      if (next < .015) continue;
      for (const [dx, dy] of directions) {
        const nx = x + dx, ny = y + dy;
        if (!open(world, nx, ny)) continue;
        const ni = index(nx, ny);
        if (field[ni] >= next) continue;
        field[ni] = next;
        queue[tail++] = ni;
      }
    }
    world.homeRevision = world.packedRevision;
  }
  refreshFoodSources(world);
  for (let i = 0; i < world.foodField.length; i++) world.foodField[i] *= .88;
  for (const food of world.exposedFood) {
    const fi = index(food.x, food.y);
    world.foodField[fi] = Math.max(world.foodField[fi], .95);
  }
  // A small diffusion pass lets an agent follow a smell beyond its current cell.
  const previous = world.foodField.slice();
  for (let y = GROUND - 1; y < ROWS - 1; y++) for (let x = 1; x < COLS - 1; x++) {
    if (!open(world, x, y)) continue;
    let strongest = previous[index(x, y)];
    for (const [dx, dy] of directions) {
      if (open(world, x + dx, y + dy)) strongest = Math.max(strongest, previous[index(x + dx, y + dy)] * .76);
    }
    world.foodField[index(x, y)] = Math.max(world.foodField[index(x, y)], strongest);
  }
}

export function sensesFor(world, ant) {
  const senses = new Float32Array(32);
  const here = index(ant.x, ant.y);
  senses[0] = ant.energy / 120;
  senses[1] = Math.min(1, ant.age / 360);
  senses[2] = ant.cargo === 'sand' ? 1 : 0;
  senses[3] = world.exposedFood.some(food => food.amount >= 1 && near(food, ant)) ? 1 : 0;
  for (let d = 0; d < 4; d++) {
    const [dx, dy] = directions[d], x = ant.x + dx, y = ant.y + dy;
    const cell = cellAt(world, x, y), at = inside(x, y) ? index(x, y) : -1;
    senses[4 + d] = world.exposedFood.some(food => food.x === x && food.y === y && food.amount >= 1) ? 1 : 0;
    senses[8 + d] = open(world, x, y) ? 1 : 0;
    senses[12 + d] = cell === PACKED ? 1 : 0;
    senses[16 + d] = at >= 0 ? world.foodField[at] : 0;
    senses[20 + d] = at >= 0 ? world.homeField[at] : 0;
  }
  senses[24] = Math.min(1, (world.occupancy?.[here] ?? 0) / 8);
  senses[25] = world.foodField[here];
  senses[26] = world.homeField[here];
  senses[27] = directions.some(([dx, dy]) => cellAt(world, ant.x + dx, ant.y + dy) === LOOSE) ? 1 : 0;
  senses[28] = Math.min(1, world.ants.length / world.populationLimit);
  senses[29] = ant.y === GROUND - 1 ? 1 : 0;
  senses[30] = world.exposedFood.some(food => food.amount >= 1
    && Math.abs(food.x - ant.x) + Math.abs(food.y - ant.y) <= 4) ? 1 : 0;
  senses[31] = 1;
  return senses;
}

const distanceTo = (x, y, target) => Math.abs(x - target.x) + Math.abs(y - target.y);
const homeTarget = ant => ({ x: ENTRANCE, y: ant.y < GROUND ? GROUND - 1 : HOME_Y });
const atHome = ant => distanceTo(ant.x, ant.y, homeTarget(ant)) <= 6;

function nearestExposedFood(world, ant, surface) {
  let best = null, distance = Infinity;
  for (const food of world.exposedFood) {
    if (food.amount < 1 || (food.y < GROUND) !== surface) continue;
    const next = distanceTo(ant.x, ant.y, food);
    if (next < distance) { best = food; distance = next; }
  }
  return distance <= 60 + ant.traits.sensing * 120 ? best : null;
}

function forageGoal(world, ant) {
  const surface = ant.traits.foraging >= .5;
  if (surface && ant.y >= GROUND) {
    if (Math.abs(ant.x - ENTRANCE) > 3 && ant.y >= HOME_Y - 20)
      return { x: ENTRANCE, y: HOME_Y };
    return { x: ENTRANCE, y: GROUND - 1 };
  }
  if (!surface && ant.y < GROUND) return Math.abs(ant.x - ENTRANCE) > 3
    ? { x: ENTRANCE, y: GROUND - 1 } : { x: ENTRANCE, y: HOME_Y };
  return nearestExposedFood(world, ant, surface);
}

function buriedFoodDistance(world, x, y) {
  let distance = Infinity;
  for (const food of world.buriedFood) {
    if (food.amount < 1) continue;
    distance = Math.min(distance, distanceTo(x, y, food));
  }
  return distance;
}

export function actionMotivation(world, ant) {
  const weights = new Float32Array(ACTIONS.length).fill(1);
  const target = ant.foodCargo ? homeTarget(ant) : ant.cargo === 'sand'
    ? ant.y > GROUND + 10
      ? { x: ant.x < ENTRANCE ? ENTRANCE - 27 : ENTRANCE + 27, y: HOME_Y }
      : { x: ant.x < ENTRANCE ? ENTRANCE - 65 : ENTRANCE + 65, y: GROUND - 1 }
    : ant.energy < 108 ? forageGoal(world, ant) : null;
  const currentGoal = target ? distanceTo(ant.x, ant.y, target) : 0;
  const currentSignal = signalPreference(world, ant, ant.bodyX, ant.bodyY);
  const currentBuriedFood = !ant.cargo && !ant.foodCargo ? buriedFoodDistance(world, ant.x, ant.y) : Infinity;
  for (let d = 0; d < 4; d++) {
    const [dx, dy] = directions[d], x = ant.x + dx, y = ant.y + dy;
    if (!canEnter(world, ant, x, y) || world.occupancy[index(x, y)] >= 12) weights[d] = .001;
    else {
      let bias = target ? (distanceTo(x, y, target) < currentGoal ? 2.8 : .48) : 1;
      const signalDelta = signalPreference(world, ant, (x + .5) * CELL, (y + .5) * CELL) - currentSignal;
      bias *= Math.exp(clamp(signalDelta * 2, -.8, .8));
      if (!ant.foodCargo && !ant.cargo) {
        const here = world.foodField[index(ant.x, ant.y)], there = world.foodField[index(x, y)];
        bias *= Math.exp(clamp((there - here) * 2.5, -.6, .6));
      }
      weights[d] = bias * (.7 + ant.traits.mobility * .6);
    }
    const digX = x, digY = y;
    const foodDistance = currentBuriedFood < Infinity ? buriedFoodDistance(world, digX, digY) : Infinity;
    const closerToFood = foodDistance < currentBuriedFood;
    weights[4 + d] = !ant.cargo && !ant.foodCargo && cellAt(world, digX, digY) === PACKED
      ? (closerToFood && (ant.y > GROUND + 4 || foodDistance <= 10)
        ? (foodDistance <= 10 ? 4 : 2) * (.5 + ant.traits.digging)
        : .06 * (.5 + ant.traits.digging)) : .001;
  }
  weights[8] = ant.cargo === 'sand' ? (ant.y <= GROUND || Math.abs(ant.x - ENTRANCE) >= 22
    || ant.cargoAge > 8 ? 5 : .08) : .001;
  weights[9] = ant.foodCargo < 2 && ant.energy < 110 && world.exposedFood.some(food => food.amount >= 1
    && near(food, ant)) ? 4 + ant.traits.foraging * 8 : .001;
  weights[10] = ant.energy > 30 && world.ants.some(other => other !== ant
    && other.energy < ant.energy - 10 && near(other, ant)) ? .2 + ant.traits.sociability * 1.2 : .001;
  weights[11] = .15 + ant.traits.secretion * .4; weights[12] = .15 + ant.traits.secretion * .4;
  weights[13] = ant.energy >= 64 + 12 * (1 - ant.traits.fecundity) && ant.harvests >= 2
    && world.time - ant.lastBirth >= 45 + 30 * (1 - ant.traits.fecundity)
    && world.ants.length < world.populationLimit ? 2 : .001;
  weights[14] = .3;
  weights[15] = target ? .5 : 1.2;
  return weights;
}

function move(world, ant, direction) {
  const [dx, dy] = directions[direction], x = ant.x + dx, y = ant.y + dy;
  if (!canEnter(world, ant, x, y) || world.occupancy[index(x, y)] >= 12) return false;
  const impulse = (8 + ant.traits.mobility * 8) * (1.1 - ant.traits.size * .2);
  ant.vx = clamp(ant.vx + dx * impulse, -24, 24);
  ant.vy = clamp(ant.vy + dy * impulse, -24, 24);
  ant.energy -= .02 + ant.traits.mobility * .03;
  return true;
}

function integrateAnt(world, ant, dt) {
  ant.prevBodyX = ant.bodyX; ant.prevBodyY = ant.bodyY;
  const drag = Math.exp(-4 * dt);
  ant.vx *= drag; ant.vy *= drag;
  for (const axis of ['x', 'y']) {
    const position = axis === 'x' ? 'bodyX' : 'bodyY';
    const velocity = axis === 'x' ? 'vx' : 'vy';
    const target = ant[position] + ant[velocity] * dt;
    const x = axis === 'x' ? Math.floor(target / CELL) : ant.x;
    const y = axis === 'y' ? Math.floor(target / CELL) : ant.y;
    if (!canEnter(world, ant, x, y) || ((x !== ant.x || y !== ant.y) && world.occupancy[index(x, y)] >= 12)) {
      ant[velocity] *= -.15;
      continue;
    }
    if (x !== ant.x || y !== ant.y) {
      world.occupancy[index(ant.x, ant.y)]--;
      world.occupancy[index(x, y)]++;
      ant.x = x; ant.y = y;
    }
    ant[position] = target;
  }
}

function spawnReactionFood(world, x, y, kind = 'bloom') {
  if (!open(world, x, y) || world.foods.length >= 96
      || world.foods.some(food => distanceTo(x, y, food) < 5)) return false;
  world.foods.push({ x, y, amount: kind === 'carcass' ? 9 : 6,
    max: kind === 'carcass' ? 9 : 18, regen: kind === 'carcass' ? 0 : .08,
    kind, age: 0 });
  if (kind === 'bloom') world.blooms++;
  return true;
}

function processEncounters(world) {
  const buckets = new Map(), key = (x, y) => `${x},${y}`;
  for (const ant of world.ants) {
    const x = Math.floor(ant.x / 4), y = Math.floor(ant.y / 4);
    const at = key(x, y);
    if (!buckets.has(at)) buckets.set(at, []);
    buckets.get(at).push(ant);
  }
  for (const ant of world.ants) {
    if (ant.encounterCooldown > world.time) continue;
    const bx = Math.floor(ant.x / 4), by = Math.floor(ant.y / 4);
    let reacted = false;
    for (let dy = -1; dy <= 1 && !reacted; dy++) for (let dx = -1; dx <= 1 && !reacted; dx++) {
      for (const other of buckets.get(key(bx + dx, by + dy)) || []) {
        if (other.id <= ant.id || other.encounterCooldown > world.time
            || distanceTo(ant.x, ant.y, other) > 3) continue;
        const towardOther = chemicalResponse(ant, other);
        const towardAnt = chemicalResponse(other, ant);
        const social = (ant.traits.sociability + other.traits.sociability) * .5;
        if (towardOther > .16 && towardAnt > .16 && social > .35) {
          world.chemicalEncounters++; world.cooperations++;
          const donor = ant.energy > other.energy ? ant : other;
          const recipient = donor === ant ? other : ant;
          const transfer = donor.energy > 40 ? Math.min(3 * social, (donor.energy - recipient.energy) * .15) : 0;
          if (transfer > .5) { donor.energy -= transfer; recipient.energy += transfer; }
          emitSignals(world, ant.bodyX, ant.bodyY, ant.signature, .8 + social);
          if (Math.random() < .018 && spawnReactionFood(world, ant.x, ant.y))
            recordEvent(world, 'bloom', ant, { partner: other.id, otherX: other.x, otherY: other.y });
          else if (Math.random() < .08)
            recordEvent(world, 'cooperate', ant, { partner: other.id, otherX: other.x, otherY: other.y });
          ant.encounterCooldown = other.encounterCooldown = world.time + 2;
          reacted = true; break;
        }
        const attacker = towardOther < -.22 && ant.traits.aggression > .58 ? ant
          : towardAnt < -.22 && other.traits.aggression > .58 ? other : null;
        if (!attacker) continue;
        const victim = attacker === ant ? other : ant;
        const damage = 1 + attacker.traits.aggression * 2;
        victim.energy -= damage; attacker.energy = Math.min(120, attacker.energy + damage * .35);
        world.chemicalEncounters++; world.conflicts++;
        emitSignals(world, victim.bodyX, victim.bodyY, attacker.signature, 1.2);
        if (Math.random() < .15) recordEvent(world, 'conflict', attacker,
          { target: victim.id, otherX: victim.x, otherY: victim.y });
        ant.encounterCooldown = other.encounterCooldown = world.time + 3;
        reacted = true; break;
      }
    }
  }
}

export function performAction(world, ant, action) {
  ant.lastAction = ant.action;
  ant.action = action;
  world.attempts[action]++;
  ant.energy -= .025;
  const succeed = () => { world.actions[action]++; return true; };
  if (action < 4) return move(world, ant, action) ? succeed() : false;
  if (action < 8) {
    if (ant.cargo || ant.foodCargo) return false;
    const [dx, dy] = directions[action - 4], x = ant.x + dx, y = ant.y + dy;
    if (!inside(x, y) || y < GROUND || cellAt(world, x, y) !== PACKED) return false;
    const grain = takeGrain(world, x, y);
    if (!grain) return false;
    refreshFoodSources(world);
    world.excavated++; ant.cargo = 'sand'; ant.cargoGrain = grain.id; ant.cargoAge = 0;
    ant.energy -= .1 + .2 * (1 - ant.traits.digging);
    recordEvent(world, 'dig', ant, { grain: grain.id });
    return succeed();
  }
  if (action === 8) {
    if (ant.cargo !== 'sand') return false;
    const candidates = directions.map(([dx, dy]) => [ant.x + dx, ant.y + dy])
      .sort((a, b) => (world.homeField[index(a[0], a[1])] || 0) - (world.homeField[index(b[0], b[1])] || 0));
    for (const [x, y] of candidates) {
      if (!inside(x, y) || !open(world, x, y) || world.occupancy[index(x, y)]) continue;
      const grain = world.particles[ant.cargoGrain];
      if (!dropGrain(world, grain, x, y, (Math.random() - .5) * 12, -8)) continue;
      world.spoiled++; ant.cargo = null; ant.cargoGrain = null; ant.cargoAge = 0;
      recordEvent(world, 'drop', ant, { grain: grain.id }); return succeed();
    }
    return false;
  }
  if (action === 9) {
    const food = world.exposedFood.find(item => item.amount >= 1 && near(item, ant));
    if (!food || ant.foodCargo >= 2 || ant.energy >= 110) return false;
    food.amount--; ant.energy = Math.min(120, ant.energy + 10 + ant.traits.foraging * 8);
    ant.foodCargo++; ant.harvests++; world.delivered++;
    world.foodField[index(ant.x, ant.y)] = 1;
    recordEvent(world, 'harvest', ant, { energy: Math.round(ant.energy) });
    return succeed();
  }
  if (action === 10) {
    if (ant.energy < 30) return false;
    const other = world.ants.find(other => other !== ant && other.energy < ant.energy - 10 && near(other, ant));
    if (!other) return false;
    ant.energy -= 8; other.energy = Math.min(120, other.energy + 7);
    recordEvent(world, 'share', ant, { recipient: other.id });
    return succeed();
  }
  if (action === 11) { world.foodField[index(ant.x, ant.y)] = Math.min(1, world.foodField[index(ant.x, ant.y)] + .6); emitSignals(world, ant.bodyX, ant.bodyY, ant.signature, .6); return succeed(); }
  if (action === 12) { world.homeField[index(ant.x, ant.y)] = Math.min(1, world.homeField[index(ant.x, ant.y)] + .4); emitSignals(world, ant.bodyX, ant.bodyY, ant.signature, .6); return succeed(); }
  if (action === 13) {
    if (ant.energy < 64 + 12 * (1 - ant.traits.fecundity) || ant.harvests < 2
        || world.time - ant.lastBirth < 45 + 30 * (1 - ant.traits.fecundity)
        || world.ants.length >= world.populationLimit) return false;
    const spots = [[ant.x, ant.y], ...directions.map(([dx, dy]) => [ant.x + dx, ant.y + dy])];
    const spot = spots.find(([x, y]) => open(world, x, y) && world.occupancy[index(x, y)] < 12);
    if (!spot) return false;
    ant.energy -= 38 + 12 * (1 - ant.traits.fecundity); ant.lastBirth = world.time;
    const child = makeCreature(world, spot[0], spot[1], ant);
    world.ants.push(child); world.occupancy[index(child.x, child.y)]++;
    world.births++;
    if (child.mutation.macro || child.mutation.bursts) world.mutationBursts++;
    recordEvent(world, 'birth', child, { parent: ant.id, color: child.color,
      neuralWeights: child.mutation.neuralWeights, traitChanges: child.mutation.traitChanges,
      bursts: child.mutation.bursts + (child.mutation.macro ? 1 : 0),
      mutationMagnitude: child.mutation.magnitude, changes: child.mutation.changes });
    return succeed();
  }
  if (action === 14) return succeed();
  if (action === 15) {
    const options = directions.map(([dx, dy]) => [ant.x + dx, ant.y + dy])
      .filter(([x, y]) => canEnter(world, ant, x, y) && world.occupancy[index(x, y)] < 12);
    if (options.length) {
      const [x, y] = options[Math.floor(Math.random() * options.length)];
      return move(world, ant, directions.findIndex(([dx, dy]) => ant.x + dx === x && ant.y + dy === y)) ? succeed() : false;
    }
  }
  return false;
}

export function stepEvolvedWorld(world, dt = .05) {
  dt = Math.max(.001, Math.min(.05, dt));
  world.time += dt; world.fieldClock += dt; world.regenClock += dt; world.settleClock += dt;
  world.snapshotClock += dt;
  world.occupancy.fill(0);
  for (const ant of world.ants) world.occupancy[index(ant.x, ant.y)]++;
  for (const ant of world.ants) integrateAnt(world, ant, dt);
  for (const ant of world.ants) {
    if (atHome(ant) && ant.foodCargo) {
      const accepted = Math.min(ant.foodCargo, MAX_STORED_FOOD - world.foodStored);
      world.foodStored += accepted; world.foodDelivered += ant.foodCargo;
      world.foodWasted += ant.foodCargo - accepted;
      recordEvent(world, 'store', ant, { food: accepted, overflow: ant.foodCargo - accepted });
      ant.foodCargo = 0;
    }
    if (ant.energy < 48 && world.foodStored > 0 && world.homeField[index(ant.x, ant.y)] > .03) {
      const wasCritical = ant.energy < 30;
      world.foodStored--; world.foodConsumed++; ant.energy = Math.min(120, ant.energy + 13);
      if (wasCritical) recordEvent(world, 'feed', ant);
    }
  }
  if (world.fieldClock >= .5) {
    updateFields(world); processEncounters(world); stepSignals(world); world.fieldClock = 0;
  }
  stepGrains(world, dt);
  if (world.regenClock >= 1) {
    for (const food of world.foods) {
      food.amount = Math.min(food.max, food.amount + food.regen);
      if (food.kind) { food.age++; food.amount = Math.max(0, food.amount - (food.kind === 'carcass' ? .06 : .02)); }
    }
    world.foods = world.foods.filter(food => !food.kind || (food.amount > .2 && food.age < 240));
    world.regenClock = 0;
  }
  const currentPopulation = world.ants.length;
  for (let i = 0; i < currentPopulation; i++) {
    const ant = world.ants[i];
    ant.age += dt; ant.energy -= dt * (.04 + .06 * ant.traits.metabolism)
      * (.8 + .4 * ant.traits.size); ant.clock += dt;
    if (ant.cargo === 'sand') ant.cargoAge += dt;
    if (ant.clock < .33) continue;
    ant.clock -= .33;
    performAction(world, ant, chooseAction(ant.brain, sensesFor(world, ant), Math.random,
      actionMotivation(world, ant)));
  }
  const survivors = [];
  for (const ant of world.ants) {
    if (ant.energy > 0 && ant.age < 420 + 240 * ant.traits.longevity) { survivors.push(ant); continue; }
    world.deaths++;
    recordEvent(world, 'death', ant, { cause: ant.energy <= 0 ? 'energy' : 'age', age: Math.round(ant.age) });
    spawnReactionFood(world, ant.x, ant.y, 'carcass');
    if (ant.cargo === 'sand') {
      const grain = world.particles[ant.cargoGrain];
      const spot = directions.map(([dx, dy]) => [ant.x + dx, ant.y + dy])
        .find(([x, y]) => open(world, x, y) && world.occupancy[index(x, y)] === 0);
      if (spot) dropGrain(world, grain, spot[0], spot[1]);
      else grain.kind = 0;
    }
  }
  world.ants = survivors;
  if (world.snapshotClock >= 10) { recordSnapshot(world); world.snapshotClock = 0; }
}

export function editSand(world, tool, from, to = from) {
  const ax = Math.floor(from.x / CELL), ay = Math.floor(from.y / CELL);
  const bx = Math.floor(to.x / CELL), by = Math.floor(to.y / CELL);
  if (!inside(bx, by) || by < 2) return false;
  if (tool === 'remove') return removeGrain(world, bx, by);
  if (tool === 'add') return Boolean(addGrain(world, bx, by));
  if (tool !== 'move' || !inside(ax, ay) || (ax === bx && ay === by) || cellAt(world, bx, by) === PACKED) return false;
  const grain = takeGrain(world, ax, ay);
  if (!grain) return false;
  if (!dropGrain(world, grain, bx, by)) { dropGrain(world, grain, ax, ay); return false; }
  return true;
}

export function paintSand(world, tool, from, to = from) {
  if (tool !== 'add' && tool !== 'remove') return false;
  const dx = to.x - from.x, dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  const steps = Math.min(512, Math.max(1, Math.ceil(distance / CELL)));
  let changed = false;
  for (let i = 0; i <= steps; i++) {
    const x = from.x + dx * i / steps, y = from.y + dy * i / steps;
    if (tool === 'add') {
      const travelX = distance ? dx / distance * 9 : 0;
      const travelY = distance ? dy / distance * 9 : 0;
      changed = Boolean(pourGrain(world, x, y, travelX, travelY)) || changed;
      for (let j = 0; j < 2; j++) {
        const angle = Math.random() * Math.PI * 2, radius = Math.sqrt(Math.random()) * 3.4;
        const px = x + Math.cos(angle) * radius, py = y + Math.sin(angle) * radius;
        changed = Boolean(pourGrain(world, px, py, travelX + Math.cos(angle) * 4,
          travelY + Math.sin(angle) * 4)) || changed;
      }
    } else {
      const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL);
      for (let oy = -2; oy <= 2; oy++) for (let ox = -2; ox <= 2; ox++) {
        if (ox * ox + oy * oy > 5) continue;
        changed = removeGrain(world, cx + ox, cy + oy) || changed;
      }
    }
  }
  return changed;
}
