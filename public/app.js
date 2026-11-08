import { WIDTH, HEIGHT, CELL, GROUND, ENTRANCE, makeEvolvedWorld, stepEvolvedWorld, cellAt, editSand, paintSand } from './evolution.js';
import { HOME_Y } from './granular.js';
import { ACTIONS, PARAMETERS } from './evolved-brain.js';
import { createVisualField } from './visual-field.js';
import { TRAIT_NAMES } from './traits.js';
import { clampCamera, panCamera, zoomCamera, MIN_ZOOM, MAX_ZOOM } from './camera.js';
import { createPondShader } from './pond-shader.js';
import { createCulture } from './culture.js';
import { createMicrobialWorld } from './microbes.js';

const $ = id => document.getElementById(id);
const art = $('art'), overlay = $('overlay'), ctx = art.getContext('2d'), octx = overlay.getContext('2d');
const pond = createPondShader($('pond'));
if (!pond) $('pond').hidden = true;
$('mode').innerHTML = `<i></i> ${pond ? 'GPU CULTURE' : 'CANVAS FALLBACK'}`;
const packedCanvas = document.createElement('canvas');
const packedCtx = packedCanvas.getContext('2d');
let renderScale = 1;
let visualField = createVisualField(), visualTick = 0, shownEventRevision = -1, shownMicrobeEventRevision = -1;
let workerCount = 180, speed = 1, paused = false, tool = 'move', world = null, culture = null, microbes = null, selectedAnt = null, selectedMicrobe = null;
let last = 0, accumulator = 0, cultureClock = 0, frames = 0, soilRevision = -1, packedRevision = -1, dragStart = null, pointer = null;
const glass = document.querySelector('.glass');
let camera = { zoom: 1, x: 0, y: 0 }, spaceHeld = false, panDrag = null, pinch = null, touchGesture = false;
const touches = new Map();
function viewport() { const { width, height } = glass.getBoundingClientRect(); return { width, height }; }
function showCamera() {
  const transform = `translate(${camera.x}px, ${camera.y}px) scale(${camera.zoom})`;
  $('pond').style.transform = art.style.transform = overlay.style.transform = transform;
  $('zoomLevel').textContent = `${Math.round(camera.zoom * 100)}%`;
  $('zoomOut').disabled = camera.zoom <= MIN_ZOOM;
  $('zoomIn').disabled = camera.zoom >= MAX_ZOOM;
}
function setZoom(zoom, anchorX, anchorY) {
  const { width, height } = viewport();
  camera = zoomCamera(camera, zoom, anchorX, anchorY, width, height);
  showCamera();
}
function moveCamera(dx, dy) {
  const { width, height } = viewport();
  camera = panCamera(camera, dx, dy, width, height);
  showCamera();
}
function glassPoint(event) {
  const rect = glass.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
function touchGeometry() {
  const [a, b] = [...touches.values()];
  const rect = glass.getBoundingClientRect();
  return { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top,
    distance: Math.hypot(a.x - b.x, a.y - b.y) };
}
function startPinch() {
  if (touches.size < 2) return;
  touchGesture = true;
  const geometry = touchGeometry();
  pinch = { zoom: camera.zoom, distance: geometry.distance,
    worldX: (geometry.x - camera.x) / camera.zoom,
    worldY: (geometry.y - camera.y) / camera.zoom };
  dragStart = null; pointer = null;
}
function updatePinch() {
  if (!pinch || touches.size < 2) return;
  const geometry = touchGeometry();
  const { width, height } = viewport();
  const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, pinch.zoom * geometry.distance / Math.max(1, pinch.distance)));
  camera = clampCamera({ zoom, x: geometry.x - pinch.worldX * zoom,
    y: geometry.y - pinch.worldY * zoom }, width, height);
  showCamera();
}
const help = { move: 'DRAG A GRAIN TO MOVE IT', remove: 'DRAG TO REMOVE GRAINS', add: 'DRAG TO POUR LOOSE GRAINS', inspect: 'CLICK AN ORGANISM OR AGENT TO INSPECT IT' };

function drawPacked() {
  const width = packedCanvas.width, top = Math.round(GROUND * CELL * renderScale);
  const height = packedCanvas.height - top;
  packedCtx.setTransform(1, 0, 0, 1, 0, 0);
  packedCtx.fillStyle = '#02090e'; packedCtx.fillRect(0, 0, width, top);
  packedCtx.fillStyle = '#041c24'; packedCtx.fillRect(0, top, width, height);
  const image = packedCtx.getImageData(0, top, width, height);
  const pixels = image.data;
  const palette = [[9, 19, 23], [12, 24, 28], [17, 31, 35], [23, 39, 42],
    [31, 50, 52], [43, 65, 65], [64, 85, 79], [105, 128, 109]];
  for (const grain of world.particles) {
    if (grain.kind !== 1) continue;
    const x = Math.round(grain.x * renderScale), y = Math.round(grain.y * renderScale) - top;
    const color = palette[grain.shade];
    const radius = Math.max(1, grain.radius * renderScale);
    const reach = Math.ceil(radius);
    for (let oy = -reach; oy <= reach; oy++) for (let ox = -reach; ox <= reach; ox++) {
      if (ox * ox + oy * oy > radius * radius) continue;
      const px = x + ox, py = y + oy;
      if (px < 0 || px >= width || py < 0 || py >= height) continue;
      const offset = (py * width + px) * 4;
      pixels[offset] = color[0]; pixels[offset + 1] = color[1]; pixels[offset + 2] = color[2];
    }
  }
  packedCtx.putImageData(image, 0, top);
  packedCtx.setTransform(renderScale, 0, 0, renderScale, 0, 0);
  packedCtx.strokeStyle = '#3dc8c7'; packedCtx.lineWidth = .42;
  packedCtx.beginPath();
  for (let y = GROUND; y < HEIGHT / CELL; y++) for (let x = 1; x < WIDTH / CELL - 1; x++) {
    const at = y * (WIDTH / CELL) + x;
    if (world.sand[at] === 1) continue;
    const px = x * CELL, py = y * CELL;
    if (world.sand[at - 1] === 1) { packedCtx.moveTo(px, py); packedCtx.lineTo(px, py + CELL); }
    if (world.sand[at + 1] === 1) { packedCtx.moveTo(px + CELL, py); packedCtx.lineTo(px + CELL, py + CELL); }
    if (y > GROUND && world.sand[at - WIDTH / CELL] === 1) { packedCtx.moveTo(px, py); packedCtx.lineTo(px + CELL, py); }
    if (y < HEIGHT / CELL - 1 && world.sand[at + WIDTH / CELL] === 1) { packedCtx.moveTo(px, py + CELL); packedCtx.lineTo(px + CELL, py + CELL); }
  }
  packedCtx.stroke();
  packedCtx.fillStyle = '#4cc1c4'; packedCtx.fillRect(0, GROUND * CELL - .35, WIDTH, .7);
  packedRevision = world.packedRevision;
}

function drawSoil() {
  if (pond) ctx.clearRect(0, 0, WIDTH, HEIGHT);
  else {
    if (packedRevision !== world.packedRevision) drawPacked();
    ctx.drawImage(packedCanvas, 0, 0, WIDTH, HEIGHT);
  }
  // A faint streak and a soft bead make motion legible without pixel blocks.
  ctx.lineCap = 'round'; ctx.lineWidth = 1.15;
  for (const aboveGround of [true, false]) {
    ctx.strokeStyle = aboveGround ? 'rgba(60,208,207,.32)' : 'rgba(141,218,222,.32)';
    ctx.fillStyle = aboveGround ? '#82d5d8' : '#9ed3d1';
    ctx.beginPath();
    for (const id of world.looseGrains) {
      const grain = world.particles[id];
      if ((grain.y < GROUND * CELL) !== aboveGround) continue;
      if ((grain.x - grain.prevX) ** 2 + (grain.y - grain.prevY) ** 2 < .2) continue;
      ctx.moveTo(grain.prevX, grain.prevY);
      ctx.lineTo(grain.x, grain.y);
    }
    ctx.stroke();
    ctx.beginPath();
    for (const id of world.looseGrains) {
      const grain = world.particles[id];
      if ((grain.y < GROUND * CELL) !== aboveGround) continue;
      ctx.moveTo(grain.x + grain.radius, grain.y);
      ctx.arc(grain.x, grain.y, grain.radius, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  soilRevision = world.revision;
}

function drawMicrobes() {
  const time = microbes.time;
  for (const organism of microbes.organisms) {
    const { x, y, radius: r, genome, state } = organism;
    const hue = genome.hue;
    const motion = Math.atan2(organism.vy, organism.vx || .001);
    const pulse = 1 + .06 * Math.sin(time * (2 + genome.motility) + organism.phase);
    octx.save(); octx.translate(x, y); octx.rotate(motion);
    if (organism.flash > 0 || organism.hit > 0 || organism === selectedMicrobe
        || (selectedMicrobe && organism.lineage === selectedMicrobe.lineage)) {
      octx.strokeStyle = organism.hit > 0 ? `hsla(4 100% 65% / ${organism.hit * .8})`
        : organism === selectedMicrobe ? '#fff1aa'
          : selectedMicrobe && organism.lineage === selectedMicrobe.lineage ? `hsla(${hue} 100% 82% / .65)`
          : `hsla(${hue} 100% 75% / ${organism.flash * .55})`;
      octx.lineWidth = organism === selectedMicrobe ? 1.2 : .8;
      octx.beginPath(); octx.arc(0, 0, r + 3 + organism.flash * 3, 0, Math.PI * 2); octx.stroke();
    }
    octx.beginPath();
    for (let j = 0; j <= 18; j++) {
      const angle = j / 18 * Math.PI * 2;
      const lobes = genome.morphology === 0 ? .13 : genome.morphology === 1 ? .065 : .035;
      const edge = r * pulse * (1 + lobes * Math.sin(angle * (genome.morphology === 0 ? 5 : 8)
        + time * 2 + organism.phase) + .035 * Math.sin(angle * 11 - time * 1.5));
      const px = Math.cos(angle) * edge, py = Math.sin(angle) * edge;
      if (j === 0) octx.moveTo(px, py); else octx.lineTo(px, py);
    }
    octx.closePath();
    octx.fillStyle = `hsla(${hue} 74% 36% / .52)`;
    octx.strokeStyle = `hsla(${hue} 100% ${state === 'fight' ? 86 : 68}% / .95)`;
    octx.lineWidth = state === 'fight' ? 1.15 : .8;
    octx.fill(); octx.stroke();
    octx.fillStyle = `hsla(${hue} 100% 71% / .22)`;
    octx.beginPath(); octx.ellipse(-r * .12, r * .04, r * .68, r * .53, .2, 0, Math.PI * 2); octx.fill();
    octx.fillStyle = state === 'divide' ? '#fff6c2' : `hsl(${hue} 100% 78%)`;
    octx.beginPath(); octx.ellipse(r * .14, 0, r * .29, r * .24, -.3, 0, Math.PI * 2); octx.fill();
    if (state === 'divide') {
      octx.beginPath(); octx.ellipse(-r * .28, 0, r * .21, r * .19, .3, 0, Math.PI * 2); octx.fill();
    }
    octx.fillStyle = '#e8fff0';
    for (let j = 0; j < 4; j++) {
      const angle = organism.phase + j * 2.4;
      octx.beginPath(); octx.arc(Math.cos(angle) * r * .48, Math.sin(angle) * r * .47,
        .42 + .28 * Math.sin(time * 2 + j + organism.phase), 0, Math.PI * 2); octx.fill();
    }
    if (genome.morphology === 1 || state === 'hunt' || state === 'fight') {
      octx.strokeStyle = `hsla(${hue} 100% 74% / .65)`;
      octx.lineWidth = .42; octx.beginPath();
      const count = state === 'hunt' || state === 'fight' ? 10 : 16;
      for (let j = 0; j < count; j++) {
        const angle = j / count * Math.PI * 2;
        const from = r * 1.02, to = r * (1.18 + .15 * Math.sin(time * 8 + j + organism.phase));
        octx.moveTo(Math.cos(angle) * from, Math.sin(angle) * from);
        octx.lineTo(Math.cos(angle) * to, Math.sin(angle) * to);
      }
      octx.stroke();
    }
    if (state === 'gather') {
      octx.fillStyle = '#fff0ab';
      octx.beginPath(); octx.arc(r * .68, -r * .42, 1.2 + organism.flash, 0, Math.PI * 2); octx.fill();
    }
    octx.restore();
  }
  for (const event of microbes.events.slice(-35)) {
    const age = time - event.time;
    if (age < 0 || age > 1.4 || !['attack', 'kill', 'divide', 'death'].includes(event.type)) continue;
    const alpha = 1 - age / 1.4;
    octx.strokeStyle = event.type === 'attack' || event.type === 'kill'
      ? `rgba(255,87,116,${alpha * .8})` : `rgba(244,232,154,${alpha * .7})`;
    octx.lineWidth = .8; octx.beginPath();
    octx.arc(event.x, event.y, 4 + age * 9, 0, Math.PI * 2); octx.stroke();
    if (event.type === 'attack' && event.targetX != null) {
      octx.beginPath(); octx.moveTo(event.x, event.y);
      octx.lineTo(event.targetX, event.targetY); octx.stroke();
    }
  }
}

function drawOverlay() {
  octx.clearRect(0, 0, WIDTH, HEIGHT);
  if (!pond) {
    octx.save();
    octx.globalCompositeOperation = 'screen';
    octx.globalAlpha = .22;
    octx.drawImage(visualField.canvas, 0, 0, WIDTH, HEIGHT);
    octx.restore();
  }
  if (!pond && culture) {
    octx.fillStyle = 'rgba(94,235,215,.35)';
    for (let i = 0; i < culture.count; i += 3)
      octx.fillRect(culture.vertices[i * 12 + 6], culture.vertices[i * 12 + 7], .75, .75);
  }
  const xNest = ENTRANCE * CELL + CELL / 2, yNest = HOME_Y * CELL + CELL / 2;
  octx.strokeStyle = '#7cf0d3'; octx.lineWidth = .55;
  octx.beginPath(); octx.arc(xNest, yNest, 6 + Math.min(8, world.foodStored / 35), 0, Math.PI * 2); octx.stroke();
  octx.fillStyle = '#d2ffeb'; octx.fillRect(xNest - 1, yNest - 1, 2, 2);
  for (const event of world.events.slice(-100)) {
    const age = world.time - event.time;
    if (age < 0 || age > 2.5 || !['dig', 'birth', 'store', 'bloom', 'cooperate', 'conflict'].includes(event.type)) continue;
    octx.strokeStyle = event.type === 'dig' ? `rgba(73,224,232,${.75 * (1 - age / 2.5)})`
      : event.type === 'conflict' ? `rgba(255,77,95,${.8 * (1 - age / 2.5)})`
      : event.type === 'birth' ? `hsla(${event.color} 98% 72% / ${.9 * (1 - age / 2.5)})`
      : `rgba(244,185,124,${.8 * (1 - age / 2.5)})`;
    octx.lineWidth = .45; octx.beginPath();
    octx.arc((event.x + .5) * CELL, (event.y + .5) * CELL, 2 + age * 3, 0, Math.PI * 2); octx.stroke();
    if (event.otherX != null) {
      octx.beginPath(); octx.moveTo((event.x + .5) * CELL, (event.y + .5) * CELL);
      octx.lineTo((event.otherX + .5) * CELL, (event.otherY + .5) * CELL); octx.stroke();
    }
  }
  for (const food of world.foods) {
    if (food.amount <= 0) continue;
    const x = food.x * CELL + CELL / 2, y = food.y * CELL + CELL / 2;
    if (world.sand[food.y * (WIDTH / CELL) + food.x] === 1) {
      octx.strokeStyle = '#3db9a5'; octx.lineWidth = .45;
      octx.beginPath(); octx.moveTo(x - 3, y); octx.lineTo(x + 3, y);
      octx.moveTo(x, y - 3); octx.lineTo(x, y + 3); octx.stroke();
      octx.fillStyle = '#b3ffe0'; octx.fillRect(x - .65, y - .65, 1.3, 1.3);
      continue;
    }
    octx.fillStyle = food.kind === 'carcass' ? '#ed876f' : food.kind === 'bloom' ? '#d8a4ff' : '#8affd9';
    octx.beginPath(); octx.moveTo(x, y - 2.2); octx.lineTo(x + 2.2, y);
    octx.lineTo(x, y + 2.2); octx.lineTo(x - 2.2, y); octx.closePath(); octx.fill();
  }
  drawMicrobes();
  for (const ant of world.ants) {
    const x = ant.bodyX, y = ant.bodyY;
    if (selectedAnt && ant.lineage === selectedAnt.lineage) {
      octx.strokeStyle = `hsla(${ant.color} 100% 78% / .9)`; octx.lineWidth = .5;
      octx.beginPath(); octx.arc(x, y, 3.2, 0, Math.PI * 2); octx.stroke();
    }
    if ((x - ant.prevBodyX) ** 2 + (y - ant.prevBodyY) ** 2 > .04) {
      octx.strokeStyle = `hsla(${ant.color} 95% 74% / .55)`;
      octx.lineWidth = .42; octx.beginPath();
      octx.moveTo(ant.prevBodyX, ant.prevBodyY); octx.lineTo(x, y); octx.stroke();
    }
    const heading = Math.atan2(ant.vy, ant.vx || .01), size = 1.2 + ant.traits.size * .7;
    octx.save(); octx.translate(x, y); octx.rotate(heading);
    octx.strokeStyle = `hsl(${ant.color} 90% 78%)`; octx.lineWidth = .34;
    octx.beginPath();
    for (const leg of [-.8, 0, .8]) {
      octx.moveTo(leg, -.25); octx.lineTo(leg - .4, -1.45);
      octx.moveTo(leg, .25); octx.lineTo(leg - .4, 1.45);
    }
    octx.stroke();
    octx.fillStyle = `hsl(${ant.color} 96% 67%)`;
    octx.beginPath(); octx.ellipse(-size * .55, 0, size * .65, .7, 0, 0, Math.PI * 2); octx.fill();
    octx.fillRect(-.25, -.42, .7, .84);
    octx.fillStyle = '#effff4'; octx.beginPath(); octx.arc(size * .65, 0, .62, 0, Math.PI * 2); octx.fill();
    if (ant.cargo === 'sand' || ant.foodCargo) {
      octx.fillStyle = ant.cargo === 'sand' ? '#f7bd79' : '#8affe0';
      octx.fillRect(size + .15, -.55, 1.1, 1.1);
    }
    octx.restore();
    if (ant === selectedAnt) { octx.strokeStyle = '#f6c76f'; octx.lineWidth = .55; octx.beginPath(); octx.arc(x, y, 4, 0, Math.PI * 2); octx.stroke(); }
  }
  if (dragStart && pointer && tool === 'move') {
    octx.strokeStyle = '#d87550'; octx.lineWidth = 2; octx.setLineDash([5, 4]);
    octx.beginPath(); octx.moveTo(dragStart.x, dragStart.y); octx.lineTo(pointer.x, pointer.y); octx.stroke(); octx.setLineDash([]);
  }
}

function updateNumbers() {
  const microbial = microbes.summary();
  const families = new Map();
  for (const ant of world.ants) {
    const family = families.get(ant.lineage) || { count: 0, generation: 0, representative: ant };
    family.count++; family.generation = Math.max(family.generation, ant.generation);
    if (ant.generation > family.representative.generation) family.representative = ant;
    families.set(ant.lineage, family);
  }
  const lineages = new Set(families.keys());
  const generation = world.ants.reduce((highest, ant) => Math.max(highest, ant.generation), 0);
  $('brood').textContent = generation;
  $('capacity').textContent = lineages.size;
  $('excavated').textContent = world.excavated;
  $('spoil').textContent = world.spoiled;
  $('workersCount').textContent = world.ants.length;
  $('foodStored').textContent = world.delivered;
  $('storedFood').textContent = world.foodStored;
  $('reactions').textContent = world.chemicalEncounters;
  $('mutations').textContent = world.mutationBursts;
  $('cultureCount').textContent = `${Math.round(culture.count / 1000)}K`;
  $('cultureBirths').textContent = culture.births;
  $('cultureMutations').textContent = culture.macroMutations;
  $('microbeCount').textContent = microbial.living;
  $('microbeLiving').textContent = microbial.living;
  $('microbeCaptures').textContent = microbial.captures;
  $('microbeBattles').textContent = microbial.attacks;
  $('microbeDivisions').textContent = microbial.divisions;
  $('microbeLargest').textContent = microbial.largest;
  $('microbeDeaths').textContent = microbial.deaths;
  $('microbeMutations').textContent = microbial.macroMutations;
  $('microbeMood').textContent = microbial.states.hunt + microbial.states.fight > 10 ? 'CONTESTED' : 'GROWING';
  $('microbeActivity').textContent = `${microbial.states.seek} SEEKING · ${microbial.states.gather} GATHERING · ${microbial.states.hunt + microbial.states.fight} HUNTING / FIGHTING · ${microbial.states.rest} RESTING`;
  const microFamilies = new Map();
  for (const organism of microbes.organisms) {
    const family = microFamilies.get(organism.lineage) || { count: 0, representative: organism, generation: 0 };
    family.count++;
    if (organism.generation >= family.generation) {
      family.generation = organism.generation; family.representative = organism;
    }
    microFamilies.set(organism.lineage, family);
  }
  $('microbeFamilies').innerHTML = [...microFamilies.entries()].sort((a, b) => b[1].count - a[1].count)
    .slice(0, 5).map(([id, family]) => `<button type="button" class="family-row" data-micro-family="${id}"><i style="background:hsl(${family.representative.genome.hue} 100% 68%)"></i><span>FAMILY ${id}</span><strong>${family.count}</strong><small>G${family.generation}</small></button>`).join('');
  $('births').textContent = microbial.divisions;
  $('deaths').textContent = microbial.deaths;
  $('taskCounts').textContent = `SUCCESSFUL MOVES ${[0, 1, 2, 3].reduce((sum, i) => sum + world.actions[i], 0)} · DIGS ${world.excavated} · BIRTHS ${world.births}/${world.attempts[13]} ATTEMPTS · FOOD DELIVERED ${world.foodDelivered}`;
  $('time').textContent = `${String(Math.floor(world.time / 60)).padStart(2, '0')}:${String(Math.floor(world.time % 60)).padStart(2, '0')}`;
  $('nestMood').textContent = world.ants.length ? 'EVOLVING' : 'EXTINCT';
  $('nestNote').textContent = `${world.ants.length} living agents from ${lineages.size} founder lineages; population ceiling ${world.populationLimit}. Generation ${generation} is the deepest surviving branch. Colored fields influence movement; food is carried to the nest.`;
  $('lineageBoard').innerHTML = [...families.entries()].sort((a, b) => b[1].count - a[1].count)
    .slice(0, 6).map(([id, family]) => `<button type="button" class="family-row" data-family="${id}"><i style="background:hsl(${family.representative.color} 96% 64%)"></i><span>FAMILY ${id}</span><strong>${family.count}</strong><small>G${family.generation}</small></button>`).join('');
  $('canvasState').textContent = paused ? 'PAUSED' : microbial.living ? 'LIVING' : 'EXTINCT';
  if (shownEventRevision !== world.eventRevision || shownMicrobeEventRevision !== microbes.eventRevision) {
    const antEvents = world.events.filter(event => event.type !== 'feed').slice(-12).map(event => {
      const label = event.type === 'birth' ? `Neural #${event.agent} born to #${event.parent}`
        : event.type === 'death' ? `Neural #${event.agent} died (${event.cause})`
        : event.type === 'share' ? `Neural #${event.agent} shared food`
        : event.type === 'store' ? `Neural #${event.agent} delivered food`
        : event.type === 'bloom' ? `Neural #${event.agent} made a nutrient bloom`
        : event.type === 'conflict' ? `Neural #${event.agent} chemical conflict`
        : `Neural #${event.agent} ${event.type}`;
      return { time: event.time, label, generation: event.generation };
    });
    const microEvents = ['divide', 'kill', 'gather', 'attack', 'death', 'emerge']
      .flatMap(type => microbes.events.filter(event => event.type === type).slice(-2))
      .map(event => ({
      time: event.time, generation: event.generation,
      label: event.type === 'attack' ? `Organism #${event.id} attacked #${event.target}`
        : event.type === 'kill' ? `Organism #${event.id} consumed #${event.target}`
        : event.type === 'divide' ? `Organism #${event.id} divided → #${event.child}${event.macro ? ' · large mutation' : ''}`
        : event.type === 'death' ? `Organism #${event.id} died; cells released`
        : event.type === 'gather' ? `Organism #${event.id} gathered ${event.total} cells`
        : `Organism #${event.id} emerged`
    }));
    const recent = [...antEvents, ...microEvents].sort((a, b) => b.time - a.time).slice(0, 9);
    $('eventLog').innerHTML = recent.length ? recent.map(event => {
      const when = `${String(Math.floor(event.time / 60)).padStart(2, '0')}:${String(Math.floor(event.time % 60)).padStart(2, '0')}`;
      return `<div><time>${when}</time><span>${event.label}</span><small>G${event.generation}</small></div>`;
    }).join('') : 'Waiting for gathering, hunting, division, or death.';
    shownEventRevision = world.eventRevision;
    shownMicrobeEventRevision = microbes.eventRevision;
  }
  updateInspector();
}

function updateInspector() {
  if (selectedMicrobe && !microbes.organisms.includes(selectedMicrobe)) selectedMicrobe = null;
  if (selectedMicrobe) {
    const cell = selectedMicrobe, gene = cell.genome;
    $('selectedRole').textContent = `ORGANISM #${cell.id}`;
    const traits = ['motility', 'aggression', 'adhesion', 'growth', 'metabolism', 'mutation'];
    const rows = traits.map(name => `<div class="trait-row"><span>${name}</span><i><b style="width:${Math.round(gene[name] * 100)}%;background:hsl(${gene.hue} 100% 70%)"></b></i><strong>${Math.round(gene[name] * 100)}</strong></div>`).join('');
    $('brainDetails').innerHTML = `<div class="phenotype-head"><i style="background:hsl(${gene.hue} 100% 68%)"></i><span>Organism #${cell.id} · family #${cell.lineage}<br>Parent #${cell.parentId ?? 'none'} · generation ${cell.generation}</span></div><div><b>${cell.state.toUpperCase()}</b> · size ${cell.radius.toFixed(1)} · mass ${cell.mass.toFixed(1)}<br>Energy ${cell.energy.toFixed(1)} · age ${cell.age.toFixed(1)}s<br>${cell.captured} cells gathered · ${cell.wounds} wounds · ${cell.kills} kills<br>${gene.morphology === 0 ? 'amoeboid' : gene.morphology === 1 ? 'ciliate' : 'round'} membrane · chemical channel ${gene.chemistry + 1}${cell.macro ? '<br>Large mutation from parent' : ''}</div><div class="inspector-label">INHERITED TRAITS</div>${rows}<div class="inspector-label">WHAT IT IS DOING</div><div>Gather: absorbs nearby small cells and grows.<br>Hunt: seeks smaller organisms from other families.<br>Rest: slows down and conserves energy.<br>Divide: shares its body and energy with a mutated child.</div>`;
    return;
  }
  if (selectedAnt && !world.ants.includes(selectedAnt)) selectedAnt = null;
  $('selectedRole').textContent = selectedAnt ? `FAMILY ${selectedAnt.lineage}` : 'NO AGENT';
  if (!selectedAnt) { $('brainDetails').innerHTML = 'Choose <b>Inspect agent</b>, then click one to see its energy, lineage, and action probabilities.'; return; }
  const ant = selectedAnt;
  const scores = [...ant.brain.outputs];
  const total = scores.reduce((sum, score) => sum + score, 0) || 1;
  const top = scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score).slice(0, 5);
  const traitRows = TRAIT_NAMES.map(name => `<div class="trait-row"><span>${name}</span><i><b style="width:${Math.round(ant.traits[name] * 100)}%;background:hsl(${ant.color} 96% 65%)"></b></i><strong>${Math.round(ant.traits[name] * 100)}</strong></div>`).join('');
  const mutation = ant.mutation;
  const mutationText = mutation ? `${mutation.neuralWeights} neural weights changed${mutation.macro ? ' · large neural block mutation' : ''}<br>${mutation.traitChanges} visible trait/chemical changes · ${mutation.bursts} large trait jumps`
    : 'Starting founder; no parent mutation.';
  const changes = mutation?.changes.slice().sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 6)
    .map(change => `<div class="mutation-row">${change.burst ? '◆' : '·'} ${change.name}: ${change.before} → ${change.after}</div>`).join('') || '';
  const ancestors = []; let ancestor = world.familyTree.find(node => node.id === ant.id);
  while (ancestor && ancestors.length < 6) {
    ancestors.unshift(`#${ancestor.id}`);
    ancestor = world.familyTree.find(node => node.id === ancestor.parentId);
  }
  $('brainDetails').innerHTML = `<div class="phenotype-head"><i style="background:hsl(${ant.color} 96% 65%)"></i><span>Agent #${ant.id} · founder family #${ant.lineage}<br>Parent #${ant.parentId ?? 'none'} · generation ${ant.generation}</span></div><div class="ancestry">${ancestors.join(' → ')}</div><div>Energy ${Math.round(ant.energy)} / 120 · age ${Math.round(ant.age)}s · food carried ${ant.foodCargo}<br>${PARAMETERS.toLocaleString()} neural weights · ${ant.traits.foraging >= .5 ? 'surface' : 'burrow'} preference<br>Last action: ${ACTIONS[ant.action]}</div><div class="inspector-label">MUTATION FROM PARENT</div><div>${mutationText}</div>${changes}<div class="inspector-label">INHERITED TRAITS</div>${traitRows}<div class="inspector-label">CHEMISTRY</div><div>Emit ${[...ant.signature].map(value => value.toFixed(2)).join(' / ')}<br>Receive ${[...ant.receptor].map(value => value.toFixed(2)).join(' / ')}</div><div class="inspector-label">CURRENT ACTION ODDS</div>`
    + top.map(({ score, i }) => `<div class="brain-row"><span>${ACTIONS[i]}</span><div class="brain-track"><i style="width:${Math.round(score / total * 100)}%"></i></div><div class="brain-value">${Math.round(score / total * 100)}%</div></div>`).join('');
}

function restart() {
  world = makeEvolvedWorld(workerCount); selectedAnt = null; selectedMicrobe = null;
  soilRevision = -1; packedRevision = -1; accumulator = 0; cultureClock = 0;
  culture = createCulture(world, pond ? 36000 : 12000);
  microbes = createMicrobialWorld(world, culture);
  if (!pond) { visualField = createVisualField(); visualField.update(world); }
  shownEventRevision = -1; shownMicrobeEventRevision = -1;
  if (pond) pond.draw(world, 0, true, culture);
  drawSoil(); drawOverlay(); updateNumbers();
}
function resizeRender() {
  const rect = document.querySelector('.glass').getBoundingClientRect();
  renderScale = Math.min(3, Math.max(1, Math.max(rect.width / WIDTH, rect.height / HEIGHT)
    * Math.min(window.devicePixelRatio || 1, 2)));
  const width = Math.ceil(WIDTH * renderScale), height = Math.ceil(HEIGHT * renderScale);
  for (const canvas of [art, overlay]) { canvas.width = width; canvas.height = height; }
  if (!pond) { packedCanvas.width = width; packedCanvas.height = height; }
  if (pond) pond.resize(width, height);
  ctx.setTransform(renderScale, 0, 0, renderScale, 0, 0);
  octx.setTransform(renderScale, 0, 0, renderScale, 0, 0);
  packedRevision = -1; soilRevision = -1;
  camera = clampCamera(camera, rect.width, rect.height); showCamera();
  if (world) { if (pond) pond.draw(world, world.time, true, culture); drawSoil(); drawOverlay(); }
}
resizeRender();
window.addEventListener('resize', resizeRender);
restart();
function frame(timestamp) {
  if (!last) last = timestamp;
  const elapsed = Math.min(.1, (timestamp - last) / 1000) * speed;
  if (!paused) {
    accumulator += elapsed;
    let steps = 0;
    while (accumulator >= .05 && steps < 8) { stepEvolvedWorld(world, .05); accumulator -= .05; steps++; }
    if (steps === 8) accumulator = 0;
    cultureClock += elapsed;
    if (frames % 2 === 0) {
      culture.step(cultureClock, microbes.organisms);
      microbes.step(cultureClock);
      cultureClock = 0;
    }
  }
  last = timestamp;
  if (soilRevision !== world.revision) drawSoil();
  if (++visualTick % 3 === 0 && !paused && !pond) visualField.update(world);
  if (pond && !paused && visualTick % 2 === 0) pond.draw(world, world.time, visualTick % 6 === 0, culture);
  drawOverlay();
  if (++frames % 10 === 0) updateNumbers();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function point(event) {
  const rect = overlay.getBoundingClientRect();
  const scale = Math.max(rect.width / WIDTH, rect.height / HEIGHT);
  const left = rect.left + (rect.width - WIDTH * scale) / 2;
  const top = rect.top + (rect.height - HEIGHT * scale) / 2;
  return { x: (event.clientX - left) / scale, y: (event.clientY - top) / scale };
}
function setPanel(open) {
  $('panel').hidden = !open;
  $('settingsToggle').setAttribute('aria-expanded', String(open));
  $('settingsToggle').setAttribute('aria-label', open ? 'Close controls' : 'Open controls');
}
$('settingsToggle').addEventListener('click', () => setPanel($('panel').hidden));
$('lineageBoard').addEventListener('click', event => {
  const button = event.target.closest('[data-family]');
  if (!button) return;
  selectedMicrobe = null;
  selectedAnt = world.ants.find(ant => ant.lineage === Number(button.dataset.family)) || null;
  updateInspector(); drawOverlay();
});
$('microbeFamilies').addEventListener('click', event => {
  const button = event.target.closest('[data-micro-family]');
  if (!button) return;
  selectedMicrobe = microbes.organisms.filter(organism => organism.lineage === Number(button.dataset.microFamily))
    .sort((a, b) => b.generation - a.generation)[0] || null;
  selectedAnt = null; updateInspector(); drawOverlay();
});
$('closePanel').addEventListener('click', () => setPanel(false));
document.addEventListener('keydown', event => { if (event.key === 'Escape') setPanel(false); });
document.addEventListener('keydown', event => {
  if (event.code !== 'Space' || (event.target instanceof Element && event.target.closest('button, input, textarea, select'))) return;
  event.preventDefault(); spaceHeld = true; overlay.classList.add('pan-ready');
});
document.addEventListener('keyup', event => {
  if (event.code !== 'Space') return;
  spaceHeld = false; overlay.classList.remove('pan-ready');
});
window.addEventListener('blur', () => { spaceHeld = false; overlay.classList.remove('pan-ready'); });
$('zoomIn').addEventListener('click', () => { const { width, height } = viewport(); setZoom(camera.zoom * 1.5, width / 2, height / 2); });
$('zoomOut').addEventListener('click', () => { const { width, height } = viewport(); setZoom(camera.zoom / 1.5, width / 2, height / 2); });
$('zoomLevel').addEventListener('click', () => { camera = { zoom: 1, x: 0, y: 0 }; showCamera(); });
overlay.addEventListener('wheel', event => {
  event.preventDefault();
  const anchor = glassPoint(event);
  setZoom(camera.zoom * Math.exp(-event.deltaY * (event.deltaMode === 1 ? .045 : .0015)), anchor.x, anchor.y);
}, { passive: false });
overlay.addEventListener('contextmenu', event => event.preventDefault());
overlay.addEventListener('pointerdown', event => {
  if (event.pointerType === 'touch') {
    touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (touches.size >= 2) { overlay.setPointerCapture(event.pointerId); startPinch(); return; }
    if (touchGesture) { overlay.setPointerCapture(event.pointerId); return; }
  }
  if (event.button === 1 || (spaceHeld && event.pointerType !== 'touch')) {
    event.preventDefault(); overlay.setPointerCapture(event.pointerId);
    panDrag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    overlay.classList.add('panning'); dragStart = null; pointer = null; return;
  }
  if (event.button !== 0) return;
  overlay.setPointerCapture(event.pointerId); dragStart = point(event); pointer = dragStart;
  if (tool === 'inspect') {
    let chosen = null, distance = Infinity;
    for (const organism of microbes.organisms) {
      const d = Math.hypot(organism.x - dragStart.x, organism.y - dragStart.y);
      if (d < organism.radius + 5 && d < distance) { chosen = organism; distance = d; }
    }
    if (chosen) {
      selectedMicrobe = chosen; selectedAnt = null;
      setPanel(true); updateInspector(); return;
    }
    selectedMicrobe = null;
    let best = null, bestDistance = 16;
    for (const ant of world.ants) {
      const d = Math.hypot(ant.bodyX - dragStart.x, ant.bodyY - dragStart.y);
      if (d < bestDistance) { bestDistance = d; best = ant; }
    }
    selectedAnt = best; if (best) setPanel(true); updateInspector();
  } else if (tool === 'add' || tool === 'remove') paintSand(world, tool, dragStart);
  else if (tool !== 'move') editSand(world, tool, dragStart);
});
overlay.addEventListener('pointermove', event => {
  if (event.pointerType === 'touch' && touches.has(event.pointerId)) {
    touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (touches.size >= 2) { updatePinch(); return; }
    if (touchGesture) return;
  }
  if (panDrag?.id === event.pointerId) {
    moveCamera(event.clientX - panDrag.x, event.clientY - panDrag.y);
    panDrag.x = event.clientX; panDrag.y = event.clientY; return;
  }
  if (!overlay.hasPointerCapture(event.pointerId)) return;
  if (pinch) return;
  const previous = pointer;
  pointer = point(event);
  if (tool === 'add' || tool === 'remove') paintSand(world, tool, previous ?? pointer, pointer);
});
overlay.addEventListener('pointerup', event => {
  if (event.pointerType === 'touch') {
    touches.delete(event.pointerId);
    if (touchGesture) {
      pinch = null; dragStart = null; pointer = null;
      if (touches.size === 0) touchGesture = false;
      return;
    }
  }
  if (panDrag?.id === event.pointerId) { panDrag = null; overlay.classList.remove('panning'); return; }
  if (tool === 'move' && dragStart) editSand(world, tool, dragStart, point(event));
  if ((tool === 'add' || tool === 'remove') && pointer) paintSand(world, tool, pointer, point(event));
  dragStart = null; pointer = null;
});
overlay.addEventListener('pointercancel', event => {
  touches.delete(event.pointerId); pinch = null;
  if (touches.size === 0) touchGesture = false;
  if (panDrag?.id === event.pointerId) { panDrag = null; overlay.classList.remove('panning'); }
  dragStart = null; pointer = null;
});
document.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click', () => {
  tool = button.dataset.tool; $('toolHelp').textContent = help[tool];
  document.querySelectorAll('[data-tool]').forEach(item => item.classList.toggle('selected', item === button));
}));
$('population').addEventListener('input', event => { $('popValue').textContent = event.target.value; });
$('population').addEventListener('change', event => { workerCount = Number(event.target.value); restart(); });
$('speed').addEventListener('input', event => { speed = Number(event.target.value); $('speedValue').textContent = `${speed}×`; });
$('reset').addEventListener('click', restart);
$('pause').addEventListener('click', () => {
  paused = !paused; $('pause').innerHTML = paused ? 'Resume world <span>▶</span>' : 'Pause world <span>Ⅱ</span>'; updateNumbers();
});
function image() {
  const output = document.createElement('canvas'); output.width = art.width; output.height = art.height;
  const context = output.getContext('2d');
  if (pond) context.drawImage(pond.canvas, 0, 0);
  context.drawImage(art, 0, 0); context.drawImage(overlay, 0, 0);
  return output;
}
$('download').addEventListener('click', () => {
  const link = document.createElement('a'); link.download = 'art-farm-black-sand.png';
  link.href = image().toDataURL('image/png'); link.click();
});
$('downloadLog').addEventListener('click', () => {
  const report = {
    format: 'art-farm-run-v1', simulatedSeconds: Math.round(world.time * 100) / 100,
    population: world.ants.length, births: world.births, deaths: world.deaths,
    deepestGeneration: world.ants.reduce((max, ant) => Math.max(max, ant.generation), 0),
    actions: Object.fromEntries(ACTIONS.map((name, i) => [name, world.actions[i]])),
    attempts: Object.fromEntries(ACTIONS.map((name, i) => [name, world.attempts[i]])),
    harvests: world.delivered, foodDelivered: world.foodDelivered, foodStored: world.foodStored,
    foodConsumed: world.foodConsumed, foodWasted: world.foodWasted,
    chemicalEncounters: world.chemicalEncounters, cooperations: world.cooperations,
    conflicts: world.conflicts, blooms: world.blooms, mutationBursts: world.mutationBursts,
    culture: culture.summary(),
    microbialEcology: { ...microbes.summary(), livingOrganisms: microbes.organisms.map(organism => ({
      id: organism.id, parentId: organism.parentId, lineage: organism.lineage,
      generation: organism.generation, x: organism.x, y: organism.y,
      mass: organism.mass, energy: organism.energy, state: organism.state,
      captured: organism.captured, kills: organism.kills, genome: organism.genome })),
      ancestry: microbes.ancestry, snapshots: microbes.snapshots,
      eventsRetained: microbes.events.length, eventLimit: 500, events: microbes.events },
    grainsDug: world.excavated, grainsDropped: world.spoiled,
    livingAgents: world.ants.map(ant => ({ id: ant.id, parentId: ant.parentId,
      founderFamily: ant.lineage, generation: ant.generation, energy: +ant.energy.toFixed(2),
      traits: ant.traits, emission: [...ant.signature], reception: [...ant.receptor],
      color: ant.color, mutation: ant.mutation })),
    familyTree: world.familyTree,
    eventsRetained: world.events.length, eventLimit: 1500, events: world.events,
    snapshotsEverySeconds: 10, snapshots: world.snapshots
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url;
  link.download = `art-farm-run-${Date.now()}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('critique').addEventListener('click', async () => {
  const button = $('critique'); button.disabled = true; $('critiqueResult').textContent = 'Jev is looking at the nest…';
  try {
    const thumb = document.createElement('canvas'); thumb.width = 800; thumb.height = 520;
    thumb.getContext('2d').drawImage(image(), 0, 0, 800, 520);
    const response = await fetch('/api/critique', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: thumb.toDataURL('image/jpeg', .82) }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Jev request failed.');
    $('jevBadge').textContent = 'CONNECTED';
    $('critiqueResult').innerHTML = Object.entries(result.scores).map(([key, value]) =>
      `<div class="score"><span>${key.toUpperCase()}</span><strong>${Math.round(value * 100)}%</strong></div>`).join('')
      + '<p>Jev observed the nest. Agent actions and reproduction remain local.</p>';
  } catch (error) { $('critiqueResult').textContent = error.message; } finally { button.disabled = false; }
});
fetch('/api/status').then(response => response.json()).then(({ jevConfigured }) => {
  if (jevConfigured) { $('mode').innerHTML = '<i></i> JEV READY'; $('jevBadge').textContent = 'READY'; }
}).catch(() => {});
