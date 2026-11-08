import { WIDTH, HEIGHT, CELL } from './evolution.js';

const SCALE = 8, COLS = Math.ceil(WIDTH / SCALE), ROWS = Math.ceil(HEIGHT / SCALE);
const CHANNELS = 4, FIELD_SIZE = COLS * ROWS * CHANNELS;
const BASE_COLORS = [
  [.08, .98, .83], [.98, .20, .61], [.65, .36, 1], [.72, 1, .18]
];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function createCulture(world, count = 36000, random = Math.random) {
  const x = new Float32Array(count), y = new Float32Array(count);
  const tailX = new Float32Array(count), tailY = new Float32Array(count);
  const vx = new Float32Array(count), vy = new Float32Array(count);
  const energy = new Float32Array(count), age = new Float32Array(count);
  const speed = new Float32Array(count), turn = new Float32Array(count);
  const channel = new Uint8Array(count), target = new Uint8Array(count);
  const receptor = new Float32Array(count * CHANNELS);
  const color = new Float32Array(count * 3);
  const vertices = new Float32Array(count * 2 * 6);
  const bucketHeads = new Int32Array(COLS * ROWS), bucketNext = new Int32Array(count);
  const taken = new Uint8Array(count);
  const lureX = new Float32Array(COLS * ROWS), lureY = new Float32Array(COLS * ROWS);
  let field = new Float32Array(FIELD_SIZE), nextField = new Float32Array(FIELD_SIZE);
  const deposits = new Float32Array(FIELD_SIZE);
  let driftX = new Float32Array(COLS * ROWS), driftY = new Float32Array(COLS * ROWS);
  let driftCount = new Uint16Array(COLS * ROWS);
  let nextDriftX = new Float32Array(COLS * ROWS), nextDriftY = new Float32Array(COLS * ROWS);
  let nextDriftCount = new Uint16Array(COLS * ROWS);
  let elapsed = 0, ticks = 0, births = 0, deaths = 0, macroMutations = 0, harvested = 0;

  function tint(index) {
    const seed = channel[index], base = BASE_COLORS[seed];
    const variation = receptor[index * CHANNELS + ((seed + 1) % CHANNELS)] * .13;
    const offset = index * 3;
    color[offset] = clamp(base[0] + variation, .05, 1);
    color[offset + 1] = clamp(base[1] - variation * .55, .05, 1);
    color[offset + 2] = clamp(base[2] + variation * .35, .05, 1);
  }

  function record(index, previousX, previousY) {
    const at = index * 12, rgb = index * 3;
    vertices[at] = previousX; vertices[at + 1] = previousY;
    vertices[at + 2] = color[rgb]; vertices[at + 3] = color[rgb + 1];
    vertices[at + 4] = color[rgb + 2]; vertices[at + 5] = .025;
    vertices[at + 6] = x[index]; vertices[at + 7] = y[index];
    vertices[at + 8] = color[rgb]; vertices[at + 9] = color[rgb + 1];
    vertices[at + 10] = color[rgb + 2]; vertices[at + 11] = .28;
  }

  function spawn(index, parent = -1) {
    const foodCount = world.foods.length;
    if (parent < 0) {
      const nearFood = foodCount && random() < .82;
      const food = nearFood ? world.foods[Math.floor(random() * foodCount)] : null;
      x[index] = clamp(food ? (food.x + .5) * CELL + (random() + random() + random() - 1.5) * 55
        : random() * WIDTH, 2, WIDTH - 2);
      y[index] = clamp(food ? (food.y + .5) * CELL + (random() + random() + random() - 1.5) * 55
        : random() * HEIGHT, 2, HEIGHT - 2);
      channel[index] = Math.floor(random() * CHANNELS);
      for (let c = 0; c < CHANNELS; c++) receptor[index * CHANNELS + c] =
        c === channel[index] ? .65 + random() * .7 : random() * 1.5 - .75;
      speed[index] = 19 + random() * 38;
      turn[index] = random() * 2 - 1;
      target[index] = foodCount ? Math.floor(random() * foodCount) : 0;
      age[index] = random() * 35;
    } else {
      x[index] = clamp(x[parent] + (random() - .5) * 9, 2, WIDTH - 2);
      y[index] = clamp(y[parent] + (random() - .5) * 9, 2, HEIGHT - 2);
      channel[index] = channel[parent];
      for (let c = 0; c < CHANNELS; c++) receptor[index * CHANNELS + c] =
        clamp(receptor[parent * CHANNELS + c] + (random() - .5) * .36, -1.5, 1.5);
      speed[index] = clamp(speed[parent] + (random() - .5) * 9, 10, 70);
      turn[index] = clamp(turn[parent] + (random() - .5) * .35, -1.5, 1.5);
      target[index] = target[parent];
      age[index] = 0;
      if (random() < .035) {
        channel[index] = Math.floor(random() * CHANNELS);
        for (let c = 0; c < CHANNELS; c++) receptor[index * CHANNELS + c] = random() * 3 - 1.5;
        speed[index] = 10 + random() * 60;
        macroMutations++;
      }
      births++;
    }
    const angle = random() * Math.PI * 2;
    vx[index] = Math.cos(angle) * speed[index] * .45;
    vy[index] = Math.sin(angle) * speed[index] * .45;
    energy[index] = .6 + random() * .6;
    tailX[index] = x[index]; tailY[index] = y[index];
    tint(index);
    record(index, x[index], y[index]);
  }

  for (let i = 0; i < count; i++) spawn(i);

  function rebuildBuckets() {
    bucketHeads.fill(-1);
    for (let i = 0; i < count; i++) {
      const cell = Math.floor(y[i] / SCALE) * COLS + Math.floor(x[i] / SCALE);
      bucketNext[i] = bucketHeads[cell]; bucketHeads[cell] = i;
    }
  }
  rebuildBuckets();

  function buildLures(organisms) {
    lureX.fill(0); lureY.fill(0);
    for (const organism of organisms) {
      const reach = Math.max(28, organism.radius * 3.2);
      const left = Math.max(0, Math.floor((organism.x - reach) / SCALE));
      const right = Math.min(COLS - 1, Math.ceil((organism.x + reach) / SCALE));
      const top = Math.max(0, Math.floor((organism.y - reach) / SCALE));
      const bottom = Math.min(ROWS - 1, Math.ceil((organism.y + reach) / SCALE));
      for (let cy = top; cy <= bottom; cy++) for (let cx = left; cx <= right; cx++) {
        const dx = organism.x - (cx + .5) * SCALE;
        const dy = organism.y - (cy + .5) * SCALE;
        const distance = Math.hypot(dx, dy);
        if (distance < 2 || distance >= reach) continue;
        const force = (1 - distance / reach) * (organism.state === 'rest' ? 2 : 12);
        const cell = cy * COLS + cx;
        lureX[cell] += dx / distance * force;
        lureY[cell] += dy / distance * force;
      }
    }
  }

  function consumeAt(px, py, radius, limit = 2) {
    let consumed = 0;
    const r2 = radius * radius;
    const left = Math.max(0, Math.floor((px - radius) / SCALE));
    const right = Math.min(COLS - 1, Math.floor((px + radius) / SCALE));
    const top = Math.max(0, Math.floor((py - radius) / SCALE));
    const bottom = Math.min(ROWS - 1, Math.floor((py + radius) / SCALE));
    for (let cy = top; cy <= bottom && consumed < limit; cy++)
      for (let cx = left; cx <= right && consumed < limit; cx++)
        for (let index = bucketHeads[cy * COLS + cx]; index >= 0 && consumed < limit; index = bucketNext[index]) {
          if (taken[index]) continue;
          const dx = x[index] - px, dy = y[index] - py;
          if (dx * dx + dy * dy > r2) continue;
          taken[index] = 1; consumed++; harvested++; deaths++;
          const parent = (index + 1 + Math.floor(random() * (count - 1))) % count;
          spawn(index, parent);
        }
    return consumed;
  }

  function releaseAt(px, py, amount = 12, chemistry = 0) {
    for (let n = 0; n < amount; n++) {
      const index = Math.floor(random() * count);
      spawn(index);
      channel[index] = chemistry % CHANNELS;
      receptor[index * CHANNELS + channel[index]] = .8 + random() * .5;
      x[index] = clamp(px + (random() - .5) * 24, 2, WIDTH - 2);
      y[index] = clamp(py + (random() - .5) * 24, 2, HEIGHT - 2);
      tailX[index] = x[index]; tailY[index] = y[index];
      tint(index); record(index, x[index], y[index]);
      taken[index] = 1; births++;
    }
  }

  function step(dt, organisms = []) {
    dt = clamp(dt, 0, .08);
    if (!dt) return;
    elapsed += dt; ticks++;
    taken.fill(0);
    buildLures(organisms);
    deposits.fill(0);
    nextDriftX.fill(0); nextDriftY.fill(0); nextDriftCount.fill(0);
    const foods = world.foods, foodCount = foods.length;
    const feeding = new Uint16Array(foodCount);
    for (let i = 0; i < count; i++) {
      const oldX = x[i], oldY = y[i];
      const cx = clamp(Math.floor(oldX / SCALE), 1, COLS - 2);
      const cy = clamp(Math.floor(oldY / SCALE), 1, ROWS - 2);
      const at = (cy * COLS + cx) * CHANNELS;
      const cellIndex = cy * COLS + cx;
      let gradientX = 0, gradientY = 0, chemistry = 0;
      for (let c = 0; c < CHANNELS; c++) {
        const response = receptor[i * CHANNELS + c]
          - (c === channel[i] ? Math.max(0, field[at + c] - .8) * .9 : 0);
        gradientX += response * (field[at + CHANNELS + c] - field[at - CHANNELS + c]);
        gradientY += response * (field[at + COLS * CHANNELS + c] - field[at - COLS * CHANNELS + c]);
        chemistry += response * field[at + c];
      }
      const foodIndex = foodCount ? target[i] % foodCount : 0;
      const preferred = foodCount ? foods[foodIndex] : null;
      let towardsX = 0, towardsY = 0, nutrient = 0;
      if (preferred && preferred.amount > 0) {
        const dx = (preferred.x + .5) * CELL - oldX;
        const dy = (preferred.y + .5) * CELL - oldY;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const hunger = clamp(1.2 - energy[i], .3, 1.1);
        const orbit = 22 + speed[i] * .55;
        const radial = clamp((distance - orbit) / 32, -1, 1) * hunger * 20;
        const circulation = 17 + Math.abs(turn[i]) * 13;
        const spin = (foodIndex % 2 ? 1 : -1) * (turn[i] < -.7 ? -1 : 1);
        towardsX = dx / distance * radial - dy / distance * circulation * spin;
        towardsY = dy / distance * radial + dx / distance * circulation * spin;
        if (distance < 86) {
          nutrient = (1 - distance / 86) * preferred.amount / Math.max(1, preferred.max);
          if (distance < 46) feeding[foodIndex]++;
        }
      } else if (foodCount) target[i] = Math.floor(random() * foodCount);
      const flowX = Math.sin(oldY * .027 + elapsed * .56) * 21
        + Math.cos((oldX + oldY) * .014 - elapsed * .31) * 12;
      const flowY = Math.cos(oldX * .023 - elapsed * .43) * 18
        - Math.sin((oldX - oldY) * .016 + elapsed * .38) * 10;
      const chemForce = 16 / (1 + Math.abs(chemistry) * .04);
      const drag = Math.exp(-1.25 * dt);
      vx[i] = (vx[i] + dt * (flowX + towardsX + gradientX * chemForce + lureX[cellIndex])) * drag;
      vy[i] = (vy[i] + dt * (flowY + towardsY + gradientY * chemForce + lureY[cellIndex])) * drag;
      if (driftCount[cellIndex] > 2) {
        const alignment = Math.min(.35, dt * .85);
        vx[i] += (driftX[cellIndex] / driftCount[cellIndex] - vx[i]) * alignment;
        vy[i] += (driftY[cellIndex] / driftCount[cellIndex] - vy[i]) * alignment;
      }
      const velocity = Math.hypot(vx[i], vy[i]);
      const limit = speed[i];
      if (velocity > limit) { vx[i] *= limit / velocity; vy[i] *= limit / velocity; }
      if (velocity < 5) { vx[i] += (random() - .5) * 8; vy[i] += (random() - .5) * 8; }
      x[i] += vx[i] * dt; y[i] += vy[i] * dt;
      if (x[i] < 2 || x[i] > WIDTH - 2) { x[i] = clamp(x[i], 2, WIDTH - 2); vx[i] *= -.85; }
      if (y[i] < 2 || y[i] > HEIGHT - 2) { y[i] = clamp(y[i], 2, HEIGHT - 2); vy[i] *= -.85; }
      const slot = (clamp(Math.floor(y[i] / SCALE), 0, ROWS - 1) * COLS
        + clamp(Math.floor(x[i] / SCALE), 0, COLS - 1)) * CHANNELS + channel[i];
      deposits[slot] += .16;
      const nextCell = Math.floor(y[i] / SCALE) * COLS + Math.floor(x[i] / SCALE);
      nextDriftX[nextCell] += vx[i]; nextDriftY[nextCell] += vy[i]; nextDriftCount[nextCell]++;
      age[i] += dt;
      energy[i] += dt * (nutrient * .15 + Math.max(-.4, chemistry) * .002 - .035);
      if (energy[i] <= 0 || age[i] > 44 + speed[i] * .45) {
        deaths++;
        let parent = -1, best = .35;
        for (let candidate = 0; candidate < 5; candidate++) {
          const trial = Math.floor(random() * count);
          if (trial !== i && energy[trial] > best) { best = energy[trial]; parent = trial; }
        }
        spawn(i, parent);
      } else if (energy[i] > 1.45 && random() < dt * .014) {
        const replaced = Math.floor(random() * count);
        if (replaced !== i && energy[replaced] < .55) { deaths++; spawn(replaced, i); energy[i] -= .35; }
      }
      tailX[i] += (x[i] - tailX[i]) * Math.min(1, dt * 4.2);
      tailY[i] += (y[i] - tailY[i]) * Math.min(1, dt * 4.2);
      record(i, tailX[i], tailY[i]);
    }
    for (let i = 0; i < foodCount; i++)
      if (feeding[i]) foods[i].amount = Math.max(0, foods[i].amount - feeding[i] * dt * .000075);
    for (let cell = 0; cell < FIELD_SIZE; cell += CHANNELS) {
      for (let c = 0; c < CHANNELS; c++) {
        const at = cell + c;
        const neighbors = field[at - CHANNELS] || 0;
        const right = field[at + CHANNELS] || 0;
        const above = field[at - COLS * CHANNELS] || 0;
        const below = field[at + COLS * CHANNELS] || 0;
        nextField[at] = clamp(field[at] * .83 + (neighbors + right + above + below) * .025
          + deposits[at] * .17, 0, 8);
      }
    }
    [field, nextField] = [nextField, field];
    [driftX, nextDriftX] = [nextDriftX, driftX];
    [driftY, nextDriftY] = [nextDriftY, driftY];
    [driftCount, nextDriftCount] = [nextDriftCount, driftCount];
    rebuildBuckets();
    if (ticks % 12 === 0 && world.hormones) {
      const signalCols = 250, signalRows = 163;
      for (let cy = 1; cy < ROWS - 1; cy += 2) for (let cx = 1; cx < COLS - 1; cx += 2) {
        const source = (cy * COLS + cx) * CHANNELS;
        const destination = ((cy * 2) * signalCols + cx * 2) * CHANNELS;
        if (cy * 2 >= signalRows || cx * 2 >= signalCols) continue;
        for (let c = 0; c < CHANNELS; c++) world.hormones[destination + c] =
          Math.min(5, world.hormones[destination + c] + field[source + c] * .012);
      }
    }
  }

  function summary() {
    const channels = [0, 0, 0, 0], samples = [];
    for (let i = 0; i < count; i++) {
      channels[channel[i]]++;
      if (i % Math.max(1, Math.floor(count / 64)) === 0 && samples.length < 64)
        samples.push({ x: +x[i].toFixed(1), y: +y[i].toFixed(1), channel: channel[i],
          receptors: [...receptor.slice(i * CHANNELS, i * CHANNELS + CHANNELS)]
            .map(value => +value.toFixed(2)), speed: +speed[i].toFixed(1),
          turn: +turn[i].toFixed(2), energy: +energy[i].toFixed(2) });
    }
    return { count, births, deaths, macroMutations, harvested, channels, genomeSamples: samples };
  }

  return { count, vertices, step, consumeAt, releaseAt, summary, get births() { return births; },
    get deaths() { return deaths; }, get macroMutations() { return macroMutations; },
    get activity() { return elapsed; } };
}
