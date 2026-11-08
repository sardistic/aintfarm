// Cohesive packed grains hold the nest; disturbed grains move freely in continuous
// space and resolve local contacts. The 2 px buckets are for searching, not slots.
export const WIDTH = 1000, HEIGHT = 650, CELL = 2;
export const COLS = WIDTH / CELL, ROWS = HEIGHT / CELL;
export const GROUND = 85, ENTRANCE = 250, HOME_Y = 165;
export const AIR = 0, PACKED = 1, LOOSE = 2;
export const GRAIN_RADIUS = .95;
const MAX_LOOSE = 16000;

const index = (x, y) => y * COLS + x;
const inside = (x, y) => x >= 0 && x < COLS && y >= 0 && y < ROWS;
const cellOf = value => Math.floor(value / CELL);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const packedAt = (world, x, y) => {
  const cx = cellOf(x), cy = cellOf(y);
  return !inside(cx, cy) || world.sand[index(cx, cy)] === PACKED;
};

export function cellAt(world, x, y) {
  if (!inside(x, y)) return PACKED;
  const at = index(x, y);
  if (world.sand[at] === PACKED) return PACKED;
  for (let id = world.bucketHead[at]; id >= 0; id = world.bucketNext[id])
    if (world.particles[id].kind === LOOSE) return LOOSE;
  return AIR;
}

function particle(world, x, y, kind, vx = 0, vy = 0) {
  const id = world.particles.length;
  let variation = Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(id, 0xc2b2ae3d);
  variation = Math.imul(variation ^ variation >>> 16, 0x7feb352d);
  variation = Math.imul(variation ^ variation >>> 15, 0x846ca68b);
  variation = (variation ^ variation >>> 16) >>> 0;
  const jitterX = ((variation & 255) / 255 - .5) * 1.6;
  const jitterY = (((variation >>> 8) & 255) / 255 - .5) * 1.6;
  const grain = { id, x: (x + .5) * CELL + (kind === PACKED ? jitterX : 0),
    y: (y + .5) * CELL + (kind === PACKED ? jitterY : 0),
    prevX: (x + .5) * CELL, prevY: (y + .5) * CELL,
    safeX: (x + .5) * CELL, safeY: (y + .5) * CELL,
    vx, vy, kind, sleep: 0, shade: variation & 7,
    radius: .7 + ((variation >>> 16) & 255) / 255 * .25 };
  world.particles.push(grain);
  if (kind === PACKED) {
    world.grainAt[index(x, y)] = id;
    world.sand[index(x, y)] = PACKED;
  } else if (kind === LOOSE) {
    world.looseGrains.add(id);
    world.activeGrains.add(id);
  }
  return grain;
}

export function makeTerrain() {
  const world = { sand: new Uint8Array(COLS * ROWS),
    grainAt: new Int32Array(COLS * ROWS).fill(-1), particles: [],
    looseGrains: new Set(), activeGrains: new Set(),
    bucketHead: new Int32Array(COLS * ROWS).fill(-1), bucketNext: new Int32Array(140000), bucketsDirty: false,
    occupancy: new Uint16Array(COLS * ROWS), revision: 1, packedRevision: 1, time: 0, settleClock: 0 };
  for (let y = GROUND; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
    const shaft = x >= ENTRANCE - 3 && x <= ENTRANCE + 3 && y <= HOME_Y + 5;
    const chamber = ((x - ENTRANCE) / 33.75) ** 2 + ((y - HOME_Y) / 18) ** 2 <= 1;
    if (!shaft && !chamber) particle(world, x, y, PACKED);
  }
  return world;
}

function clearPacked(world, grain) {
  const x = cellOf(grain.x), y = cellOf(grain.y);
  const at = index(x, y);
  if (world.grainAt[at] !== grain.id) return;
  world.grainAt[at] = -1;
  world.sand[at] = AIR;
  world.packedRevision++;
}

function wakeLooseNear(world, x, y) {
  if (world.bucketsDirty) rebuildBuckets(world);
  for (let ny = y - 3; ny <= y + 3; ny++) for (let nx = x - 3; nx <= x + 3; nx++) {
    if (!inside(nx, ny)) continue;
    for (let id = world.bucketHead[index(nx, ny)]; id >= 0; id = world.bucketNext[id]) {
      if (world.particles[id].kind !== LOOSE) continue;
      world.particles[id].sleep = 0;
      world.activeGrains.add(id);
    }
  }
}

function wakeAbove(world, x, y) {
  wakeLooseNear(world, x, y);
  const aboveY = y - 1;
  if (!inside(x, aboveY)) return;
  const id = world.grainAt[index(x, aboveY)];
  if (id < 0) return;
  const above = world.particles[id];
  let bonds = 0;
  for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]])
    if (inside(x + dx, aboveY + dy) && world.sand[index(x + dx, aboveY + dy)] === PACKED) bonds++;
  if (bonds < 2) {
    clearPacked(world, above);
    above.kind = LOOSE;
    world.looseGrains.add(id);
    world.activeGrains.add(id);
    world.bucketsDirty = true;
  }
}

function nearestLoose(world, x, y) {
  if (world.bucketsDirty) rebuildBuckets(world);
  const px = (x + .5) * CELL, py = (y + .5) * CELL;
  let best = null, bestDistance = (CELL * 1.5) ** 2;
  for (let ny = y - 2; ny <= y + 2; ny++) for (let nx = x - 2; nx <= x + 2; nx++) {
    if (!inside(nx, ny)) continue;
    for (let id = world.bucketHead[index(nx, ny)]; id >= 0; id = world.bucketNext[id]) {
      const grain = world.particles[id];
      if (grain.kind !== LOOSE) continue;
      const distance = (grain.x - px) ** 2 + (grain.y - py) ** 2;
      if (distance < bestDistance) { best = grain; bestDistance = distance; }
    }
  }
  return best;
}

export function takeGrain(world, x, y) {
  if (!inside(x, y)) return null;
  const packedId = world.grainAt[index(x, y)];
  const grain = packedId >= 0 ? world.particles[packedId] : nearestLoose(world, x, y);
  if (!grain) return null;
  if (grain.kind === PACKED) { clearPacked(world, grain); wakeAbove(world, x, y); }
  else world.looseGrains.delete(grain.id);
  world.activeGrains.delete(grain.id);
  grain.kind = 3; // carried; the exact same grain is returned when dropped
  grain.vx = 0; grain.vy = 0; grain.sleep = 0;
  world.revision++;
  return grain;
}

export function dropGrain(world, grain, x, y, vx = 0, vy = 0) {
  if (!grain || !inside(x, y) || y < 2 || world.sand[index(x, y)] === PACKED
      || (world.occupancy?.[index(x, y)] ?? 0) > 0 || world.looseGrains.size >= MAX_LOOSE) return false;
  grain.x = (x + .5) * CELL; grain.y = (y + .5) * CELL;
  grain.prevX = grain.x; grain.prevY = grain.y;
  grain.safeX = grain.x; grain.safeY = grain.y;
  grain.vx = vx; grain.vy = vy; grain.kind = LOOSE; grain.sleep = 0;
  world.looseGrains.add(grain.id);
  world.activeGrains.add(grain.id);
  world.bucketsDirty = true;
  world.revision++;
  return true;
}

export function pourGrain(world, px, py, vx = 0, vy = 0) {
  const x = cellOf(px), y = cellOf(py);
  if (!inside(x, y) || y < 2 || world.sand[index(x, y)] === PACKED
      || (world.occupancy?.[index(x, y)] ?? 0) > 0 || world.looseGrains.size >= MAX_LOOSE) return null;
  const grain = particle(world, x, y, LOOSE, vx, vy);
  grain.x = px; grain.y = py; grain.prevX = px; grain.prevY = py;
  grain.safeX = px; grain.safeY = py;
  world.bucketsDirty = true;
  world.revision++;
  return grain;
}

export function addGrain(world, x, y) {
  return pourGrain(world, (x + .5) * CELL, (y + .5) * CELL);
}

export function removeGrain(world, x, y) {
  const grain = takeGrain(world, x, y);
  if (!grain) return false;
  grain.kind = 0;
  return true;
}

function ensureBucketCapacity(world) {
  if (world.bucketNext.length >= world.particles.length) return;
  world.bucketNext = new Int32Array(Math.max(world.particles.length, world.bucketNext.length * 2));
}

function rebuildBuckets(world) {
  ensureBucketCapacity(world);
  world.bucketHead.fill(-1);
  for (const id of world.looseGrains) {
    const grain = world.particles[id];
    const x = clamp(cellOf(grain.x), 0, COLS - 1);
    const y = clamp(cellOf(grain.y), 0, ROWS - 1);
    const at = index(x, y);
    world.bucketNext[id] = world.bucketHead[at];
    world.bucketHead[at] = id;
  }
  world.bucketsDirty = false;
}

function separateFromPacked(world, grain) {
  const cx = cellOf(grain.x), cy = cellOf(grain.y);
  for (let y = cy - 1; y <= cy + 1; y++) for (let x = cx - 1; x <= cx + 1; x++) {
    if (!inside(x, y)) continue;
    const id = world.grainAt[index(x, y)];
    if (id < 0) continue;
    const other = world.particles[id];
    let dx = grain.x - other.x, dy = grain.y - other.y;
    let distance = Math.hypot(dx, dy);
    const spacing = grain.radius + other.radius + .08;
    if (distance >= spacing) continue;
    if (distance < .0001) { dx = 0; dy = -1; distance = 1; }
    const overlap = spacing - distance;
    grain.x += dx / distance * overlap;
    grain.y += dy / distance * overlap;
  }
}

function separateFromLoose(world, grain) {
  const cx = clamp(cellOf(grain.x), 0, COLS - 1), cy = clamp(cellOf(grain.y), 0, ROWS - 1);
  for (let y = cy - 1; y <= cy + 1; y++) for (let x = cx - 1; x <= cx + 1; x++) {
    if (!inside(x, y)) continue;
    for (let id = world.bucketHead[index(x, y)]; id >= 0; id = world.bucketNext[id]) {
      if (id === grain.id) continue;
      const other = world.particles[id];
      let dx = grain.x - other.x, dy = grain.y - other.y;
      let distance = Math.hypot(dx, dy);
      const spacing = grain.radius + other.radius + .05;
      if (distance < .0001) { dx = id < grain.id ? 1 : -1; dy = 0; distance = 1; }
      if (distance < spacing) {
        const shift = (spacing - distance) * .52;
        grain.x += dx / distance * shift; grain.y += dy / distance * shift;
        other.x -= dx / distance * shift; other.y -= dy / distance * shift;
        if (!world.activeGrains.has(id)) {
          other.prevX = other.x; other.prevY = other.y;
          other.sleep = 0; world.activeGrains.add(id);
        }
      } else if (distance < spacing + 1.5) {
        // Weak cohesion and viscosity make poured grains flow as a dense stream.
        const pull = (distance - spacing) * .004;
        grain.x -= dx / distance * pull;
        grain.y -= dy / distance * pull;
        grain.vx += (other.vx - grain.vx) * .004;
      }
    }
  }
}

export function stepGrains(world, dt) {
  const active = [...world.activeGrains];
  if (!active.length) return;
  for (const id of active) {
    const grain = world.particles[id];
    if (grain.kind !== LOOSE) { world.activeGrains.delete(id); continue; }
    grain.prevX = grain.x; grain.prevY = grain.y;
    grain.vy = Math.min(75, grain.vy + 235 * dt);
    grain.vx *= .985;
    // Sweep against packed grains in short increments so a fast grain cannot
    // cross a one-pixel wall between two frames.
    const slices = Math.max(1, Math.ceil(Math.hypot(grain.vx, grain.vy) * dt / .65));
    for (let slice = 0; slice < slices; slice++) {
      const oldX = grain.x, oldY = grain.y;
      grain.x = clamp(grain.x + grain.vx * dt / slices, grain.radius, WIDTH - grain.radius);
      grain.y = clamp(grain.y + grain.vy * dt / slices, grain.radius, HEIGHT - grain.radius);
      if (packedAt(world, grain.x, grain.y)) {
        grain.x = oldX; grain.y = oldY;
        grain.vx += ((id & 1) ? 1 : -1) * 2.5;
        grain.vy *= -.08;
      } else {
        separateFromPacked(world, grain);
        if (packedAt(world, grain.x, grain.y)) { grain.x = oldX; grain.y = oldY; }
      }
    }
  }
  for (let pass = 0; pass < 2; pass++) {
    rebuildBuckets(world);
    for (const id of active) {
      const grain = world.particles[id];
      if (grain.kind !== LOOSE) continue;
      separateFromPacked(world, grain);
      separateFromLoose(world, grain);
      grain.x = clamp(grain.x, grain.radius, WIDTH - grain.radius);
      grain.y = clamp(grain.y, grain.radius, HEIGHT - grain.radius);
      if (packedAt(world, grain.x, grain.y)) { grain.x = grain.safeX; grain.y = grain.safeY; }
    }
  }
  for (const id of world.looseGrains) {
    const grain = world.particles[id];
    if (packedAt(world, grain.x, grain.y)) {
      grain.x = grain.safeX; grain.y = grain.safeY;
      grain.vx *= .25; grain.vy = 0;
    } else {
      grain.safeX = grain.x; grain.safeY = grain.y;
    }
  }
  rebuildBuckets(world);
  for (const id of active) {
    const grain = world.particles[id];
    if (grain.kind !== LOOSE) continue;
    grain.vx = (grain.x - grain.prevX) / dt * .8;
    grain.vy = (grain.y - grain.prevY) / dt * .8;
    const speed = Math.hypot(grain.vx, grain.vy);
    grain.sleep = speed < 3 ? grain.sleep + 1 : 0;
    if (grain.sleep > 10) world.activeGrains.delete(id);
  }
  world.revision++;
}
