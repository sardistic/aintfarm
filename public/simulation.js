import { createBrain, evaluateBrain, rewardBrain } from './brain.js';

export const WIDTH = 1000, HEIGHT = 650, CELL = 10;
export const COLS = WIDTH / CELL, ROWS = HEIGHT / CELL;
export const GROUND = 17, ENTRANCE = 50;
const AIR = 0, PACKED = 1, LOOSE = 2;
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const index = (x, y) => y * COLS + x;
const inside = (x, y) => x >= 0 && x < COLS && y >= 0 && y < ROWS;
const neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const random = (a, b) => a + Math.random() * (b - a);

export function makeWorld(workerCount = 180) {
  const sand = new Uint8Array(COLS * ROWS);
  for (let y = GROUND; y < ROWS; y++) for (let x = 0; x < COLS; x++) sand[index(x, y)] = PACKED;
  // A starter shaft and brood chamber give the colony somewhere to begin.
  for (let y = GROUND; y <= 34; y++) for (let x = ENTRANCE - 1; x <= ENTRANCE + 1; x++) sand[index(x, y)] = AIR;
  for (let y = 30; y <= 36; y++) for (let x = 44; x <= 56; x++) {
    if (((x - 50) / 6.7) ** 2 + ((y - 33) / 3.6) ** 2 <= 1) sand[index(x, y)] = AIR;
  }
  const world = { sand, ants: [], foods: [{ x: 12, amount: 90 }, { x: 28, amount: 85 },
      { x: 73, amount: 85 }, { x: 88, amount: 90 }],
    broodStages: { eggs: 8, larvae: 10, pupae: 6 }, brood: 24, nurseCare: 0,
    foodStored: 18, delivered: 0, excavated: 0, spoiled: 0, births: 0, deaths: 0, time: 0,
    workerLimit: Math.min(600, workerCount + 100), revision: 1, settleClock: 0,
    broodClock: 0, seedClock: 0, pressure: 1, branching: 1 };
  for (let i = 0; i < workerCount; i++) world.ants.push(makeAnt(i % 5 < 3 ? 'digger' : i % 5 === 3 ? 'forager' : 'nurse'));
  return world;
}

function makeAnt(role) {
  return { x: ENTRANCE + Math.round(random(-3, 3)), y: 33 + Math.round(random(-2, 2)),
    role, cargo: null, target: null, clock: random(0, .12), speed: random(.11, .17),
    turnBias: random(-.25, .25), trips: 0, careActions: 0, energy: random(75, 100), age: 0,
    lifespan: random(600, 900), brain: createBrain(role), brainClock: random(0, .8),
    foodMemory: null, searchDir: Math.random() < .5 ? -1 : 1, digProgress: 0 };
}

export function cellAt(world, x, y) { return inside(x, y) ? world.sand[index(x, y)] : PACKED; }
export function capacity(world) {
  if (world.capacityRevision === world.revision) return world.capacityCache;
  let room = 0;
  for (let y = 25; y < ROWS; y++) for (let x = 0; x < COLS; x++) if (cellAt(world, x, y) === AIR) room++;
  world.capacityCache = Math.floor(room / 5);
  world.capacityRevision = world.revision;
  return world.capacityCache;
}
export function needsDigging(world) { return capacity(world) < world.brood + Math.ceil(12 * world.pressure); }

function walkable(world, x, y) {
  return inside(x, y) && y >= GROUND - 1 && (y === GROUND - 1 || cellAt(world, x, y) === AIR);
}

// BFS follows the existing open nest; ants cannot move through packed sand.
export function nextStep(world, start, goal) {
  if (!walkable(world, start.x, start.y) || !walkable(world, goal.x, goal.y)) return null;
  const startIndex = index(start.x, start.y), goalIndex = index(goal.x, goal.y);
  if (startIndex === goalIndex) return null;
  const previous = new Int32Array(world.sand.length).fill(-1);
  const queue = new Int32Array(world.sand.length);
  let head = 0, tail = 0;
  queue[tail++] = startIndex; previous[startIndex] = startIndex;
  while (head < tail && previous[goalIndex] === -1) {
    const current = queue[head++], x = current % COLS, y = Math.floor(current / COLS);
    for (const [dx, dy] of neighbors) {
      const nx = x + dx, ny = y + dy;
      if (!walkable(world, nx, ny)) continue;
      const ni = index(nx, ny);
      if (previous[ni] !== -1) continue;
      previous[ni] = current; queue[tail++] = ni;
    }
  }
  if (previous[goalIndex] === -1) return null;
  let step = goalIndex;
  while (previous[step] !== startIndex) step = previous[step];
  return { x: step % COLS, y: Math.floor(step / COLS) };
}

function shapeScore(world, x, y) {
  const rx = Math.min(11, 6.5 + world.brood * .25 * world.pressure);
  const ry = Math.min(5.5, 3.5 + world.brood * .1 * world.pressure);
  let score = ((x - ENTRANCE) / rx) ** 2 + ((y - 33) / ry) ** 2;
  if (world.brood >= 28) {
    const left = ((x - 35) / 5.5) ** 2 + ((y - 47) / 3.5) ** 2;
    const corridor = Math.abs(x - (45 - (y - 36) * .8)) / 2.2;
    score = Math.min(score, left, y >= 35 && y <= 47 ? corridor : Infinity);
  }
  if (world.brood >= 36) {
    const right = ((x - 65) / 5.5) ** 2 + ((y - 48) / 3.5) ** 2;
    const corridor = Math.abs(x - (55 + (y - 36) * .85)) / 2.2;
    score = Math.min(score, right, y >= 35 && y <= 48 ? corridor : Infinity);
  }
  return score;
}

function targetForDigging(world, ant) {
  let best = null, score = Infinity;
  for (let y = 25; y <= 53; y++) {
    for (let x = 28; x <= 72; x++) {
      if (cellAt(world, x, y) !== PACKED) continue;
      const shape = shapeScore(world, x, y);
      if (shape > 1.05) continue;
      for (const [dx, dy] of neighbors) {
        const approach = { x: x + dx, y: y + dy };
        if (!walkable(world, approach.x, approach.y)) continue;
        const d = Math.abs(approach.x - ant.x) + Math.abs(approach.y - ant.y);
        const branching = world.branching > 1 ? Math.abs(x - 50) * (world.branching - 1) : 0;
        const branchWork = world.brood >= 28 && y > 36 ? 12 : 0;
        const candidateScore = d + shape * 2 - branching - branchWork + random(0, 1.5);
        if (candidateScore < score) { score = candidateScore; best = { sand: { x, y }, approach }; }
      }
    }
  }
  return best;
}

function moveToward(world, ant, goal) {
  const step = nextStep(world, ant, goal);
  if (step) {
    const crowd = world.occupancy?.[index(step.x, step.y)] ?? 0;
    if (Math.random() < Math.min(.5, Math.max(0, crowd - 2) * .08)) return false;
    if (world.occupancy) { world.occupancy[index(ant.x, ant.y)]--; world.occupancy[index(step.x, step.y)]++; }
    ant.x = step.x; ant.y = step.y; ant.energy = Math.max(0, ant.energy - .065);
    return true;
  }
  return false;
}

function dumpSand(world, ant) {
  // Pellets are deposited to either side of the entrance and remain in the grid.
  let best = null, bestScore = Infinity;
  for (let x = ENTRANCE - 18; x <= ENTRANCE + 18; x++) {
    const distance = Math.abs(x - ENTRANCE);
    if (distance < 5) continue;
    let top = GROUND - 1;
    while (top >= 2 && cellAt(world, x, top) === LOOSE) top--;
    if (top < 2 || cellAt(world, x, top) !== AIR) continue;
    const height = GROUND - 1 - top;
    const score = height * 5 + distance * .18 + ((x + world.spoiled * 7) % 5) * .03;
    if (score < bestScore) { bestScore = score; best = { x, y: top }; }
  }
  if (!best) return false;
  world.sand[index(best.x, best.y)] = LOOSE;
  world.spoiled++; world.revision++; ant.cargo = null; ant.target = null; ant.trips++;
  rewardBrain(ant.brain, 'digger');
  return true;
}

function diggerStep(world, ant) {
  if (ant.cargo === 'sand') {
    if (ant.x === ENTRANCE && ant.y === GROUND - 1) { dumpSand(world, ant); return; }
    moveToward(world, ant, { x: ENTRANCE, y: GROUND - 1 }); return;
  }
  if (!needsDigging(world)) { foragerStep(world, ant); return; }
  if (!ant.target || cellAt(world, ant.target.sand.x, ant.target.sand.y) !== PACKED) {
    ant.target = targetForDigging(world, ant); ant.digProgress = 0;
  }
  if (!ant.target) return;
  const { sand, approach } = ant.target;
  if (ant.x === approach.x && ant.y === approach.y) {
    ant.digProgress++;
    ant.energy = Math.max(0, ant.energy - .3);
    if (ant.digProgress < 1 + Math.floor((sand.y - GROUND) / 15)) return;
    world.sand[index(sand.x, sand.y)] = AIR;
    world.excavated++; world.revision++; ant.cargo = 'sand'; ant.target = null; ant.digProgress = 0;
    return;
  }
  moveToward(world, ant, approach);
}

function foragerStep(world, ant) {
  if (ant.cargo === 'food') {
    const brood = { x: ENTRANCE, y: 33 };
    if (ant.x === brood.x && ant.y === brood.y) {
      ant.cargo = null; ant.trips++; world.foodStored++; world.delivered++;
      rewardBrain(ant.brain, 'forager');
      for (const other of world.ants) {
        if (other === ant || other.foodMemory !== null || Math.abs(other.x - ant.x) + Math.abs(other.y - ant.y) > 5) continue;
        if (Math.random() < .22) other.foodMemory = ant.foodMemory;
      }
      return;
    }
    moveToward(world, ant, brood); return;
  }
  if (ant.y !== GROUND - 1) { moveToward(world, ant, { x: ENTRANCE, y: GROUND - 1 }); return; }
  const visible = world.foods.find(food => food.amount > 0 && Math.abs(food.x - ant.x) <= 7);
  const remembered = world.foods.find(food => food.amount > 0 && food.x === ant.foodMemory);
  const food = visible ?? remembered;
  if (food) {
    if (ant.x === food.x) { food.amount--; ant.cargo = 'food'; ant.foodMemory = food.x; return; }
    ant.searchDir = Math.sign(food.x - ant.x);
  }
  const nextX = ant.x + ant.searchDir;
  if (nextX < 2 || nextX >= COLS - 2) ant.searchDir *= -1;
  moveToward(world, ant, { x: ant.x + ant.searchDir, y: GROUND - 1 });
}

function nurseStep(world, ant) {
  const brood = { x: ENTRANCE, y: 33 };
  if (Math.abs(ant.x - brood.x) + Math.abs(ant.y - brood.y) > 3) { moveToward(world, ant, brood); return; }
  world.nurseCare += .14;
  ant.energy = Math.max(0, ant.energy - .035);
  if (ant.careActions++ % 24 === 0) rewardBrain(ant.brain, 'nurse', .4);
}

function restStep(world, ant) {
  const brood = { x: ENTRANCE, y: 33 };
  if (Math.abs(ant.x - brood.x) + Math.abs(ant.y - brood.y) > 2) { moveToward(world, ant, brood); return; }
  ant.energy = Math.min(100, ant.energy + (world.foodStored > .04 ? 2.8 : .6));
  if (world.foodStored > .04) world.foodStored -= .04;
}

function settleLoose(world) {
  let changed = false;
  for (let y = ROWS - 2; y >= 0; y--) for (let x = 1; x < COLS - 1; x++) {
    if (cellAt(world, x, y) !== LOOSE) continue;
    let nx = x;
    if (cellAt(world, x, y + 1) !== AIR) {
      if (y >= GROUND - 1) continue;
      const outward = x < ENTRANCE ? -1 : 1;
      if (Math.abs(x + outward - ENTRANCE) < 4 || cellAt(world, x + outward, y + 1) !== AIR) continue;
      nx += outward;
    }
    if (world.ants.some(ant => ant.x === nx && ant.y === y + 1)) continue;
    world.sand[index(nx, y + 1)] = LOOSE; world.sand[index(x, y)] = AIR; changed = true;
  }
  if (changed) world.revision++;
}

function sensesFor(world, ant) {
  const clamp01 = n => clamp(n, 0, 1);
  let contacts = 0;
  for (const other of world.ants) if (other !== ant && Math.abs(other.x - ant.x) + Math.abs(other.y - ant.y) <= 2) contacts++;
  const foodNearby = ant.y === GROUND - 1 && world.foods.some(food => food.amount > 0 && Math.abs(food.x - ant.x) <= 7);
  const broodDistance = Math.abs(ant.x - ENTRANCE) + Math.abs(ant.y - 33);
  const soilNearby = ant.y >= 25 && neighbors.some(([dx, dy]) => cellAt(world, ant.x + dx, ant.y + dy) === PACKED);
  return [
    clamp01((world.brood + 12 * world.pressure - capacity(world)) / 16),
    clamp01(.32 + (world.brood * .7 - world.foodStored) / 24),
    clamp01((world.broodStages.larvae * 1.2 - world.nurseCare * .1) / 16),
    clamp01(contacts / 12), 1 - ant.energy / 100,
    foodNearby || ant.foodMemory !== null ? 1 : 0,
    clamp01(1 - broodDistance / 9), soilNearby ? 1 : 0
  ];
}

function broodCycle(world) {
  const stages = world.broodStages;
  const caredFor = world.nurseCare >= Math.max(1, stages.larvae * .4);
  if (caredFor && stages.pupae > 0 && world.ants.length < world.workerLimit) {
    const born = Math.min(stages.pupae, Math.max(1, Math.floor(stages.pupae * .18)), world.workerLimit - world.ants.length);
    stages.pupae -= born; world.births += born;
    for (let i = 0; i < born; i++) world.ants.push(makeAnt('nurse'));
  }
  if (caredFor && stages.larvae > 0 && world.foodStored >= 1) {
    const grow = Math.min(stages.larvae, Math.max(1, Math.floor(stages.larvae * .18)));
    stages.larvae -= grow; stages.pupae += grow; world.foodStored -= grow * .35;
  }
  if (caredFor && stages.eggs > 0) {
    const hatch = Math.min(stages.eggs, Math.max(1, Math.floor(stages.eggs * .22)));
    stages.eggs -= hatch; stages.larvae += hatch;
  }
  if (world.foodStored >= 3 && capacity(world) > world.brood - 2) {
    stages.eggs += 3; world.foodStored -= 1.1;
  }
  world.foodStored = Math.max(0, world.foodStored - world.ants.length * .08);
  world.brood = stages.eggs + stages.larvae + stages.pupae;
  world.nurseCare = 0;
}

function dropNewFood(world) {
  world.foods = world.foods.filter(food => food.amount > 0);
  const x = Math.floor(random(5, COLS - 5));
  if (Math.abs(x - ENTRANCE) > 7 && world.foods.length < 8) world.foods.push({ x, amount: Math.round(random(25, 55)) });
}

export function stepWorld(world, dt = .05) {
  dt = clamp(dt, .001, .05); world.time += dt; world.settleClock += dt; world.broodClock += dt; world.seedClock += dt;
  if (world.settleClock >= .18) { settleLoose(world); world.settleClock = 0; }
  if (world.broodClock >= 12) { broodCycle(world); world.broodClock = 0; }
  if (world.seedClock >= 25) { dropNewFood(world); world.seedClock = 0; }
  world.occupancy = new Uint16Array(world.sand.length);
  for (const ant of world.ants) world.occupancy[index(ant.x, ant.y)]++;
  for (const ant of world.ants) {
    ant.clock += dt; ant.brainClock += dt; ant.age += dt;
    if (ant.brainClock >= .7 && ant.cargo === null) {
      const role = ant.energy < 16 ? 'rest' : evaluateBrain(ant.brain, sensesFor(world, ant), ant.role);
      if (role !== ant.role) { ant.role = role; ant.target = null; ant.digProgress = 0; }
      ant.brainClock = 0;
    }
    if (ant.clock < ant.speed) continue;
    ant.clock -= ant.speed;
    if (ant.role === 'digger') diggerStep(world, ant);
    else if (ant.role === 'forager') foragerStep(world, ant);
    else if (ant.role === 'nurse') nurseStep(world, ant);
    else restStep(world, ant);
  }
  const survivors = [];
  for (const ant of world.ants) {
    if (ant.age < ant.lifespan && ant.energy > 0) { survivors.push(ant); continue; }
    world.deaths++;
    if (ant.cargo === 'sand') {
      for (const [dx, dy] of [[0, 0], ...neighbors]) {
        const x = ant.x + dx, y = ant.y + dy;
        if (y < GROUND || cellAt(world, x, y) !== AIR) continue;
        world.sand[index(x, y)] = LOOSE; world.revision++; break;
      }
    }
  }
  world.ants = survivors;
}

export function editSand(world, tool, from, to = from) {
  const a = { x: Math.floor(from.x / CELL), y: Math.floor(from.y / CELL) };
  const b = { x: Math.floor(to.x / CELL), y: Math.floor(to.y / CELL) };
  if (!inside(a.x, a.y) || !inside(b.x, b.y) || a.y < 2 || b.y < 2) return false;
  if (tool === 'food') {
    if (b.y > GROUND - 1 || world.foods.length >= 8) return false;
    world.foods.push({ x: b.x, amount: 28 }); return true;
  }
  if (tool === 'move') {
    if (cellAt(world, a.x, a.y) === AIR || cellAt(world, b.x, b.y) !== AIR) return false;
    if (a.x === b.x && a.y === b.y) return false;
    if (world.ants.some(ant => ant.x === b.x && ant.y === b.y) || (b.x === ENTRANCE && b.y === 33)) return false;
    world.sand[index(b.x, b.y)] = LOOSE; world.sand[index(a.x, a.y)] = AIR;
  } else if (tool === 'remove') {
    if (cellAt(world, b.x, b.y) === AIR) return false;
    world.sand[index(b.x, b.y)] = AIR;
  } else if (tool === 'add') {
    if (cellAt(world, b.x, b.y) !== AIR) return false;
    if (world.ants.some(ant => ant.x === b.x && ant.y === b.y) || (b.x === ENTRANCE && b.y === 33)) return false;
    world.sand[index(b.x, b.y)] = LOOSE;
  } else return false;
  world.revision++; return true;
}

export function applyJev(world, scores) {
  world.pressure = clamp(1 + (1 - scores.chambers) * .6, 1, 1.6);
  world.branching = clamp(1 + (1 - scores.branches) * .55, 1, 1.55);
}
