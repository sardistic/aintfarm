import { WIDTH, HEIGHT, CELL } from './evolution.js';
import { signalAt, emitSignals } from './signals.js';

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const HUES = [168, 326, 267, 79, 27, 210];

export function createMicrobialWorld(world, culture, population = 84, random = Math.random) {
  const organisms = [], events = [], ancestry = [], snapshots = [];
  let nextId = 1, time = 0, snapshotClock = 0, chemicalClock = 0, captures = 0, attacks = 0, divisions = 0, deaths = 0, macroMutations = 0;
  let eventRevision = 0;

  function record(type, organism, extra = {}) {
    events.push({ type, time: +time.toFixed(2), id: organism.id,
      lineage: organism.lineage, generation: organism.generation,
      x: organism.x, y: organism.y, ...extra });
    if (events.length > 500) events.shift();
    eventRevision++;
  }

  function create(parent = null, px = null, py = null) {
    const id = nextId++;
    const food = world.foods.length ? world.foods[Math.floor(random() * world.foods.length)] : null;
    const x = clamp(px ?? (food ? (food.x + .5) * CELL + (random() - .5) * 90 : random() * WIDTH), 15, WIDTH - 15);
    const y = clamp(py ?? (food ? (food.y + .5) * CELL + (random() - .5) * 90 : random() * HEIGHT), 15, HEIGHT - 15);
    const genome = parent ? {
      motility: clamp(parent.genome.motility + (random() - .5) * .2, .05, 1),
      aggression: clamp(parent.genome.aggression + (random() - .5) * .24, 0, 1),
      adhesion: clamp(parent.genome.adhesion + (random() - .5) * .24, 0, 1),
      growth: clamp(parent.genome.growth + (random() - .5) * .2, .05, 1),
      metabolism: clamp(parent.genome.metabolism + (random() - .5) * .16, .05, 1),
      mutation: clamp(parent.genome.mutation + (random() - .5) * .14, .02, 1),
      hue: (parent.genome.hue + (random() - .5) * 23 + 360) % 360,
      morphology: parent.genome.morphology,
      chemistry: parent.genome.chemistry
    } : {
      motility: random(), aggression: random(), adhesion: random(), growth: random(),
      metabolism: random(), mutation: random(),
      hue: (HUES[(id - 1) % HUES.length] + (random() - .5) * 26 + 360) % 360,
      morphology: Math.floor(random() * 3), chemistry: Math.floor(random() * 4)
    };
    let macro = false;
    if (parent && random() < .06 + genome.mutation * .13) {
      const trait = ['motility', 'aggression', 'adhesion', 'growth', 'metabolism'][Math.floor(random() * 5)];
      genome[trait] = clamp(genome[trait] + (random() < .5 ? -.45 : .45), 0, 1);
      genome.hue = (genome.hue + (random() < .5 ? -80 : 80) + 360) % 360;
      genome.morphology = Math.floor(random() * 3);
      genome.chemistry = Math.floor(random() * 4);
      macro = true;
      macroMutations++;
    }
    const mass = parent ? Math.max(20, parent.mass * .82) : 35 + random() * 80;
    const organism = {
      id, parentId: parent?.id ?? null, lineage: parent?.lineage ?? id,
      generation: parent ? parent.generation + 1 : 0, genome, x, y,
      vx: (random() - .5) * 15, vy: (random() - .5) * 15,
      mass, radius: 3 + Math.sqrt(mass) * .72,
      energy: parent ? Math.max(24, parent.energy * .88) : 42 + random() * 50,
      age: parent ? 0 : random() * 36, phase: random() * Math.PI * 2,
      state: 'seek', captured: 0, kills: 0, wounds: 0,
      feedClock: random() * .25, attackClock: random() * .5,
      divisionClock: parent ? 12 : random() * 9,
      flash: 0, hit: 0, macro
    };
    ancestry.push({ id, parentId: organism.parentId, lineage: organism.lineage,
      generation: organism.generation, bornAt: +time.toFixed(2), genome: { ...genome }, macro });
    return organism;
  }

  for (let i = 0; i < population; i++) organisms.push(create());

  function step(dt) {
    dt = clamp(dt, 0, .08);
    if (!dt) return;
    time += dt;
    snapshotClock += dt;
    chemicalClock += dt;
    const emitThisStep = chemicalClock >= .2;
    if (emitThisStep) chemicalClock = 0;
    const initialCount = organisms.length;
    const newborns = [];
    for (let i = 0; i < initialCount; i++) {
      const organism = organisms[i];
      if (organism.energy <= 0) continue;
      const gene = organism.genome;
      organism.age += dt;
      organism.feedClock -= dt; organism.attackClock -= dt; organism.divisionClock -= dt;
      organism.flash = Math.max(0, organism.flash - dt * 3);
      organism.hit = Math.max(0, organism.hit - dt * 2);
      let prey = null, threat = null, preyDistance = Infinity, threatDistance = Infinity;
      let separateX = 0, separateY = 0;
      for (let j = 0; j < initialCount; j++) {
        const other = organisms[j];
        if (other === organism || other.energy <= 0) continue;
        const dx = other.x - organism.x, dy = other.y - organism.y;
        const distance = Math.hypot(dx, dy) || .01;
        if (distance < organism.radius + other.radius + 4) {
          const pressure = (organism.radius + other.radius + 4 - distance) / distance;
          separateX -= dx * pressure; separateY -= dy * pressure;
        }
        if (other.lineage !== organism.lineage && distance < 90) {
          if (gene.aggression > .52 && other.mass < organism.mass * 1.45 && distance < preyDistance) {
            prey = other; preyDistance = distance;
          }
          if (other.genome.aggression > .65 && other.mass > organism.mass * 1.35
              && distance < threatDistance) { threat = other; threatDistance = distance; }
        }
      }
      let directionX = 0, directionY = 0;
      const resting = organism.energy > 67 && Math.sin(time * .35 + organism.phase) > .96;
      if (threat && gene.aggression < .68 && threatDistance < 55) {
        organism.state = 'flee';
        directionX = organism.x - threat.x; directionY = organism.y - threat.y;
      } else if (prey && gene.aggression > .56 && organism.energy < 105) {
        organism.state = 'hunt';
        directionX = prey.x - organism.x; directionY = prey.y - organism.y;
      } else if (resting) {
        organism.state = 'rest';
      } else {
        organism.state = 'seek';
        let best = null, score = Infinity;
        for (const food of world.foods) {
          if (food.amount < 1) continue;
          const dx = (food.x + .5) * CELL - organism.x;
          const dy = (food.y + .5) * CELL - organism.y;
          const distance = Math.hypot(dx, dy) / Math.max(.3, food.amount / food.max);
          if (distance < score) { score = distance; best = food; }
        }
        if (best) {
          directionX = (best.x + .5) * CELL - organism.x;
          directionY = (best.y + .5) * CELL - organism.y;
          const distance = Math.hypot(directionX, directionY);
          if (distance < 34) {
            const orbitX = -directionY, orbitY = directionX;
            directionX = orbitX; directionY = orbitY;
          }
        }
      }
      const directionLength = Math.hypot(directionX, directionY) || 1;
      const movement = (8 + gene.motility * 19) / (1 + organism.radius * .014)
        * (organism.state === 'rest' ? .2 : 1);
      const signalX = signalAt(world, organism.x + 8, organism.y, gene.chemistry)
        - signalAt(world, organism.x - 8, organism.y, gene.chemistry);
      const signalY = signalAt(world, organism.x, organism.y + 8, gene.chemistry)
        - signalAt(world, organism.x, organism.y - 8, gene.chemistry);
      organism.vx += dt * ((directionX / directionLength * movement - organism.vx) * 1.6
        + separateX * 5 + signalX * gene.adhesion * 9 + Math.sin(time + organism.phase) * 3);
      organism.vy += dt * ((directionY / directionLength * movement - organism.vy) * 1.6
        + separateY * 5 + signalY * gene.adhesion * 9 + Math.cos(time * .8 + organism.phase) * 3);
      const velocity = Math.hypot(organism.vx, organism.vy);
      if (velocity > 45) { organism.vx *= 45 / velocity; organism.vy *= 45 / velocity; }
      organism.x = clamp(organism.x + organism.vx * dt, 12, WIDTH - 12);
      organism.y = clamp(organism.y + organism.vy * dt, 12, HEIGHT - 12);
      if (organism.x === 12 || organism.x === WIDTH - 12) organism.vx *= -.5;
      if (organism.y === 12 || organism.y === HEIGHT - 12) organism.vy *= -.5;
      organism.energy -= dt * (.22 + gene.metabolism * .5 + organism.radius * .018)
        * (organism.state === 'rest' ? .48 : 1);
      if (organism.feedClock <= 0 && organism.state !== 'flee') {
        const caught = culture.consumeAt(organism.x, organism.y, organism.radius + 4,
          organism.mass > 95 ? 3 : 2);
        organism.feedClock = .18 + (1 - gene.adhesion) * .17;
        if (caught) {
          organism.captured += caught; captures += caught;
          organism.energy = Math.min(125, organism.energy + caught * (1.3 + gene.growth));
          organism.mass = Math.min(1600, organism.mass + caught * (1.1 + gene.growth * 2.5));
          organism.radius = 3 + Math.sqrt(organism.mass) * .72;
          organism.state = 'gather'; organism.flash = 1;
          if (organism.captured % 24 < caught) record('gather', organism, { total: organism.captured });
        }
      }
      if (prey && prey.energy > 0 && organism.attackClock <= 0
          && preyDistance < organism.radius + prey.radius + 5 && gene.aggression > .56) {
        const damage = 2 + gene.aggression * 3.5 + Math.sqrt(organism.mass) * .16;
        prey.energy -= damage; prey.mass = Math.max(5, prey.mass - damage * .28);
        prey.radius = 3 + Math.sqrt(prey.mass) * .72;
        prey.wounds++; prey.hit = 1;
        organism.energy = Math.min(125, organism.energy + damage * .45);
        organism.mass = Math.min(1600, organism.mass + damage * .13);
        organism.radius = 3 + Math.sqrt(organism.mass) * .72;
        organism.attackClock = .42; organism.state = 'fight'; organism.flash = 1;
        attacks++; record('attack', organism, { target: prey.id, targetX: prey.x, targetY: prey.y });
        if (prey.energy <= 0) { organism.kills++; record('kill', organism, { target: prey.id }); }
      }
      if (organism.energy > 88 && organism.mass > 72 && organism.captured >= 10
          && organism.divisionClock <= 0 && organisms.length + newborns.length < 170) {
        organism.energy *= .55; organism.mass *= .57;
        organism.radius = 3 + Math.sqrt(organism.mass) * .72;
        organism.divisionClock = 17 + (1 - gene.growth) * 9;
        const child = create(organism, organism.x + (random() - .5) * 18,
          organism.y + (random() - .5) * 18);
        child.energy = organism.energy * .9;
        child.mass = organism.mass * .9;
        child.radius = 3 + Math.sqrt(child.mass) * .72;
        newborns.push(child); divisions++;
        organism.state = 'divide'; organism.flash = 1;
        record('divide', organism, { child: child.id, macro: child.macro });
      }
      if (emitThisStep) {
        const signature = [-.8, -.8, -.8, -.8];
        signature[gene.chemistry] = 1;
        signature[(gene.chemistry + 1) % 4] = gene.adhesion * 1.6 - .8;
        emitSignals(world, organism.x, organism.y, signature,
          (.12 + gene.adhesion * .18) * (organism.state === 'gather' ? 1.6 : 1));
      }
    }
    const survivors = [];
    for (const organism of organisms) {
      if (organism.energy > 0 && organism.age < 85 + organism.genome.metabolism * 70) {
        survivors.push(organism); continue;
      }
      deaths++; record('death', organism, { cause: organism.energy <= 0 ? 'energy' : 'age' });
      culture.releaseAt(organism.x, organism.y, Math.min(35, 8 + Math.floor(organism.mass / 10)),
        organism.genome.chemistry);
    }
    organisms.splice(0, organisms.length, ...survivors, ...newborns);
    if (organisms.length < 20 && world.foods.length) {
      const immigrant = create(); organisms.push(immigrant); record('emerge', immigrant);
    }
    if (snapshotClock >= 5) {
      snapshots.push({ time: +time.toFixed(1), ...summary() });
      if (snapshots.length > 300) snapshots.shift();
      snapshotClock = 0;
    }
  }

  function summary() {
    const states = { seek: 0, gather: 0, hunt: 0, fight: 0, flee: 0, rest: 0, divide: 0 };
    let largest = 0;
    for (const organism of organisms) {
      states[organism.state]++;
      largest = Math.max(largest, organism.radius);
    }
    return { living: organisms.length, captures, attacks, divisions, deaths, macroMutations,
      largest: +largest.toFixed(1), states, eventsRetained: events.length };
  }

  return { organisms, events, ancestry, snapshots, step, summary,
    get eventRevision() { return eventRevision; }, get time() { return time; } };
}
