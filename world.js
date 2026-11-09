// The shared world: one territory map everyone sees. Strains grow from their nodes like lichen, powered by the GPUs
// of viewers currently watching, and borders move according to which strains actually survive in viewers' dishes.
// Each strain's signature gene shapes how its land behaves (fast creepers, fortresses, walls, jumpers, leeches...).
// Outside forces push back while anyone is watching: nature (droughts, blooms, storms) and chaos (an evolving blight).
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { strainSignature, EFFECTS } from './public/field/genebank.js';

export const WORLD_W = 64;                  // territory cells per side (each covers 12×12 dish cells at G=768)
export const BLIGHT = 'b119b700';           // the blight's pseudo-strain tag in the owner map
const CELLS = WORLD_W * WORLD_W;
const HEX8 = /^[0-9a-f]{8}$/;
const MAX_RATE = 12e6;                      // particle-steps/s; GPU power saturates here so one card can't dominate
const OBSERVE_INTERVAL_MS = 4000;           // per session
const POWER_DECAY = 0.85;                   // per tick, once a strain has stopped reporting
const REPORT_GRACE_TICKS = 12;              // clients report every 5 s; allow a missed report before fading
const OFFLINE_EROSION = 0.004;              // strength lost per tick by cells of strains with no power...
const DORMANT_FLOOR = 0.12;                 // ...down to this: unwatched territory goes dormant, it doesn't vanish
const ABANDONED_MS = 14 * 86400_000;        // a strain nobody has watched for this long finally crumbles away
const TICK_MS = 1000;
const NT = EFFECTS.length;

// How a signature gene embodies itself in territory. s spread, a attack, d defence, br blight resistance,
// jump chance to spore past the front, plus specials.
const TRAITS = {
  1: { s: 1.3, a: 1.1, d: 0.9 },                       // Seekers
  2: { s: 1.15, a: 1.15 },                             // Vortex
  3: { d: 1.3 },                                       // Contour
  4: { bloom: 1.8, br: 0.3 },                          // Ferment
  5: { s: 1.5, d: 0.8 },                               // Darters
  6: { s: 0.7, d: 1.6 },                               // Holdfast
  7: { a: 1.4, d: 0.9 },                               // Crush
  8: { s: 1.2, br: 0.15 },                             // Pulsar
  9: { a: 1.2, leech: true },                          // Leech
  10: { s: 1.1, d: 1.1 },                              // Relay
  11: { a: 1.3, drought: 0.5 },                        // Ember: hot, shrugs off drought
  12: { d: 1.25, br: 0.75 },                           // Carapace: the blight's natural enemy
  13: { s: 1.2, jump: 0.08 },                          // Drifters
  14: { scav: true, br: -0.3 },                        // Scavenger: takes dormant land, catches viruses
  15: { jump: 0.15, br: -0.4 },                        // Spore: far-flung, but gene-swappers spread infection
  16: { colony: true },                                // Colony: stronger with kin around
  17: { s: 0.8, d: 1.7 },                              // Membrane: walls
};
const NATURE = ['drought', 'bloom', 'storm'];

export function createWorld({ dataDir, strainOf, touch = () => {} }) {
  const owner = new Array(CELLS).fill('');   // strain tag (hex) or ''
  const strength = new Float32Array(CELLS);
  const influence = Array.from({ length: CELLS }, () => new Map());   // tag -> evidence from viewers' dishes
  const power = new Map();                   // tag -> recent particle-steps/s of its viewers (EMA)
  const reported = new Map();                // tag -> sessions reporting this tick
  const lastReportTick = new Map();          // tag -> tick of its latest report
  const lastObs = new Map();                 // sid -> ms
  const sessionRate = new Map();             // sid -> { rate, tick, strain }: each viewer's live GPU contribution
  const powerHistory = [];                   // network total, sampled every 5 ticks (newest last)
  const events = [];                         // recent notable moments, newest first
  const traitCache = new Map();
  const peak = new Map();                    // tag -> most cells ever held
  const infected = new Map();                // tag -> cells the blight took this tick
  const file = join(dataDir, 'world.json');
  let tickNo = 0, dirty = false, born = Date.now();
  // nature: regional events; chaos: the blight, a virus whose genes mutate and keep what works
  let nature = [];                           // { kind, x, y, r, left }
  const natMap = new Uint8Array(CELLS);      // 0 none, 1 drought, 2 bloom, 3 storm
  let blight = freshBlight(), blightPrev = null, blightGained = 0, blightPrevScore = -1;

  const idx = (x, y) => ((y + WORLD_W) % WORLD_W) * WORLD_W + ((x + WORLD_W) % WORLD_W);
  const neighbours = c => { const x = c % WORLD_W, y = (c / WORLD_W) | 0; return [idx(x + 1, y), idx(x - 1, y), idx(x, y + 1), idx(x, y - 1)]; };
  const pw = tag => Math.sqrt(Math.min(power.get(tag) || 0, MAX_RATE) / MAX_RATE);   // 0..1
  const sigOf = tag => strainSignature(parseInt(tag, 16) >>> 0);
  function traitOf(tag) {
    if (!tag || tag === BLIGHT) return { s: 1, a: 1, d: 1, br: 0, jump: 0 };
    let t = traitCache.get(tag);
    if (!t) { t = { s: 1, a: 1, d: 1, br: 0, jump: 0, ...TRAITS[sigOf(tag)] }; traitCache.set(tag, t); }
    return t;
  }
  function note(text) { events.unshift({ t: Date.now(), text }); events.length = Math.min(events.length, 20); }
  function freshBlight() { return { gen: 0, vir: 0.25, spread: 0.08, aff: new Array(NT).fill(1) }; }

  async function load() {
    try {
      const o = JSON.parse(await readFile(file, 'utf8'));
      if (o.W !== WORLD_W) return;
      o.owner.forEach((t, i) => { if (HEX8.test(t)) owner[i] = t; });
      o.strength.forEach((s, i) => { strength[i] = Math.max(0, Math.min(1, +s || 0)); });
      if (Number.isFinite(o.born)) born = o.born;
      for (const [k, v] of Object.entries(o.peak || {})) if (HEX8.test(k) && Number.isFinite(v)) peak.set(k, v);
      if (o.blight && Array.isArray(o.blight.aff) && o.blight.aff.length === NT) {
        blight = { gen: +o.blight.gen || 0, vir: +o.blight.vir || 0.25, spread: +o.blight.spread || 0.08, aff: o.blight.aff.map(v => +v || 1) };
      }
      if (Array.isArray(o.nature)) nature = o.nature.filter(e => NATURE.includes(e?.kind)).slice(0, 3);
      console.log(`[world] loaded territory (${owner.filter(Boolean).length} cells held)`);
    } catch (e) { if (e.code !== 'ENOENT') console.warn('[world] could not load:', e.message); }
  }
  async function save() {
    if (!dirty) return;
    dirty = false;
    try {
      await mkdir(dataDir, { recursive: true });
      await writeFile(file + '.tmp', JSON.stringify({ W: WORLD_W, born, peak: Object.fromEntries(peak), blight, nature, owner,
        strength: Array.from(strength, s => +s.toFixed(3)) }));
      await rename(file + '.tmp', file);
    } catch (e) { dirty = true; console.warn('[world] could not save:', e.message); }
  }

  // A viewer's dish reports which strains it saw surviving where (sampled particles), plus its GPU throughput.
  function observe(body, ip) {
    const now = Date.now();
    const sid = typeof body?.sid === 'string' && /^[a-z0-9]{8,32}$/.test(body.sid) ? body.sid : null;
    if (!sid || !HEX8.test(body.strain || '') || body.strain === BLIGHT) return { status: 400, body: { error: 'Bad request.' } };
    if (now - (lastObs.get(sid) || 0) < OBSERVE_INTERVAL_MS) return { status: 429, body: { error: 'Too frequent.' } };
    lastObs.set(sid, now);
    touch(body.strain, body.node);
    if (lastObs.size > 20_000) for (const [k, t] of lastObs) if (now - t > 600_000) lastObs.delete(k);
    const rate = Math.min(Math.max(0, Number(body.rate) || 0), MAX_RATE * 2);
    power.set(body.strain, (power.get(body.strain) || 0) * 0.7 + rate * 0.3 * (1 + (reported.get(body.strain) || 0) * 0.5));
    reported.set(body.strain, (reported.get(body.strain) || 0) + 1);
    lastReportTick.set(body.strain, tickNo);
    sessionRate.set(sid, { rate: Math.min(rate, MAX_RATE * 2), tick: tickNo, strain: body.strain });
    // the viewer's own GPU weights its evidence; total evidence per report is capped
    const w = Math.sqrt(Math.min(rate, MAX_RATE) / MAX_RATE);
    let total = 0;
    for (const o of Array.isArray(body.obs) ? body.obs.slice(0, 600) : []) {
      if (!Array.isArray(o)) continue;
      const [c, tag, n] = o;
      if (!Number.isInteger(c) || c < 0 || c >= CELLS || !HEX8.test(tag) || !strainOf(tag)) continue;
      let k = Math.min(Math.max(0, Math.floor(n) || 0), 64);
      if (tag === body.strain) k *= 0.3;   // a dish is mostly its viewer's own strain; wins elsewhere count more
      if (!k || (total += k) > 800) continue;
      const m = influence[c];
      m.set(tag, (m.get(tag) || 0) + k * w * 0.15);
    }
    return { status: 200, body: { ok: true } };
  }

  function claim(c, tag, s) {
    const prev = owner[c];
    owner[c] = tag; strength[c] = s; dirty = true;
    return prev && prev !== tag ? prev : null;
  }

  // the whole network's live GPU power: every viewer who reported recently (hidden tabs report 0)
  function network() {
    let total = 0, viewers = 0;
    for (const [sid, s] of sessionRate) {
      if (tickNo - s.tick > REPORT_GRACE_TICKS) { sessionRate.delete(sid); continue; }
      if (s.rate > 0) { total += s.rate; viewers++; }
    }
    return { total, viewers };
  }

  // ----- nature -----
  function natureTick(active) {
    for (const e of nature) e.left--;
    nature = nature.filter(e => e.left > 0);
    if (active && nature.length < 3 && Math.random() < 1 / 90) {
      const kind = NATURE[(Math.random() * 3) | 0], x = (Math.random() * WORLD_W) | 0, y = (Math.random() * WORLD_W) | 0;
      nature.push({ kind, x, y, r: 4 + Math.round(Math.random() * 6), left: 60 + Math.round(Math.random() * 120) });
      const who = owner[idx(x, y)] || 'empty ground';
      note(kind === 'drought' ? `a drought settled over ${who}` : kind === 'bloom' ? `a bloom made ${who} fertile` : `a storm tore across ${who}`);
      dirty = true;
    }
    natMap.fill(0);
    for (const e of nature) {
      const k = NATURE.indexOf(e.kind) + 1;
      for (let dy = -e.r; dy <= e.r; dy++) for (let dx = -e.r; dx <= e.r; dx++) if (dx * dx + dy * dy <= e.r * e.r) natMap[idx(e.x + dx, e.y + dy)] = k;
    }
  }

  // ----- chaos: the blight -----
  function blightEvolve(leader) {
    // hill climbing on its own genes: keep a mutation if the blight spread at least as well, otherwise revert
    if (blightPrev && blightGained < blightPrevScore) blight = blightPrev;
    else if (blightPrev) blight.gen++;
    blightPrevScore = blightGained; blightGained = 0;
    blightPrev = { ...blight, aff: blight.aff.slice() };
    const n = () => (Math.random() + Math.random() + Math.random() - 1.5) * 0.5;
    blight = { gen: blight.gen, vir: Math.min(0.9, Math.max(0.05, blight.vir * Math.exp(n()))),
      spread: Math.min(0.3, Math.max(0.02, blight.spread * Math.exp(n()))),
      aff: blight.aff.map(v => Math.min(3, Math.max(0.2, v * Math.exp(n())))) };
    // and it adapts toward whoever is winning, so no strain stays on top for free
    if (leader) { const s = sigOf(leader); blight.aff[s] = Math.min(3, blight.aff[s] * 1.2); }
    dirty = true;
  }
  function blightCell(c, cap) {
    strength[c] -= 0.006;
    if (strength[c] <= 0) { owner[c] = ''; strength[c] = 0; dirty = true; return; }
    if (Math.random() > blight.spread) return;
    const n = neighbours(c)[(Math.random() * 4) | 0], victim = owner[n];
    if (victim === BLIGHT) return;
    if (!victim) { if (Math.random() < 0.15) { claim(n, BLIGHT, 0.3); blightGained++; } return; }
    const T = traitOf(victim);
    const chance = blight.vir * blight.aff[sigOf(victim)] * (1 - T.br) * (1 - 0.6 * strength[n]) * cap;
    if (Math.random() < chance) { claim(n, BLIGHT, 0.5); blightGained++; infected.set(victim, (infected.get(victim) || 0) + 1); }
    else strength[n] = Math.max(0.05, strength[n] - 0.03 * blight.vir);   // a setback even when it fails
  }

  function tick() {
    tickNo++;
    if (tickNo % 5 === 0) { powerHistory.push(Math.round(network().total)); if (powerHistory.length > 120) powerHistory.shift(); }
    const overrun = new Map();   // "a>b" -> cells taken this tick
    for (const [tag, p] of power) {
      if (tickNo - (lastReportTick.get(tag) ?? -1e9) > REPORT_GRACE_TICKS) power.set(tag, p * POWER_DECAY);
      if (power.get(tag) < 1000) { power.delete(tag); lastReportTick.delete(tag); }
    }
    reported.clear();
    const active = power.size > 0;           // outside forces only act while someone is watching
    natureTick(active);
    let blightCells = 0, leader = '', leadN = 0;
    const counts = new Map();
    for (const t of owner) {
      if (!t) continue;
      if (t === BLIGHT) { blightCells++; continue; }
      const k = (counts.get(t) || 0) + 1; counts.set(t, k);
      if (k > leadN) { leadN = k; leader = t; }
    }
    if (active && tickNo % 60 === 0) blightEvolve(leader);
    if (active && !blightCells && leadN > 40 && Math.random() < 1 / 300) {
      // an outbreak, most likely in the biggest territory
      const cs = []; for (let c = 0; c < CELLS; c++) if (owner[c] === leader) cs.push(c);
      claim(cs[(Math.random() * cs.length) | 0], BLIGHT, 0.7);
      note(`a blight broke out in ${leader}'s land`);
    }
    const cap = blightCells > CELLS * 0.08 ? 0.3 : 1;
    // spores: a powered strain's node cell is always seeded if free
    for (const tag of power.keys()) {
      const node = strainOf(tag)?.node;
      if (!node) continue;
      const c = idx(Math.floor(node.x * WORLD_W), Math.floor(node.y * WORLD_W));
      if (!owner[c]) claim(c, tag, 0.4);
    }
    const order = Array.from({ length: CELLS }, (_, i) => i).sort(() => Math.random() - 0.5);
    for (const c of order) {
      // 1) evidence from dishes: the strain seen most here reinforces or contests the cell
      const m = influence[c];
      let best = '', bv = 0;
      for (const [tag, v] of m) { const nv = v * 0.85; if (nv < 0.01) m.delete(tag); else { m.set(tag, nv); if (nv > bv) { bv = nv; best = tag; } } }
      const own = owner[c];
      if (best && best === own) { strength[c] = Math.min(1, strength[c] + 0.05 * Math.min(1, bv)); }
      else if (best) {
        const defend = strength[c] + (m.get(own) || 0) * 0.5;
        const frontier = neighbours(c).some(n => owner[n] === best);
        if (bv > defend && frontier) {
          const prev = claim(c, best, 0.35);
          if (prev) overrun.set(best + '>' + prev, (overrun.get(best + '>' + prev) || 0) + 1);
        } else if (frontier) strength[c] = Math.max(0, strength[c] - 0.04 * bv);
      }
      const tag = owner[c];
      if (!tag) continue;
      const nat = natMap[c];
      // storms scramble: cells weaken and sometimes swap hands at random
      if (nat === 3 && active && Math.random() < 0.03) {
        strength[c] *= 0.6;
        const n = neighbours(c)[(Math.random() * 4) | 0];
        if (owner[n] && owner[n] !== tag && Math.random() < 0.4) { owner[c] = owner[n]; owner[n] = tag; dirty = true; }
        continue;
      }
      if (tag === BLIGHT) { if (active) blightCell(c, cap); continue; }
      // 2) lichen growth: established cells of powered strains creep into their neighbours, shaped by their traits
      const p = pw(tag), T = traitOf(tag);
      if (p === 0) {
        if (strength[c] > DORMANT_FLOOR) strength[c] = Math.max(DORMANT_FLOOR, strength[c] - OFFLINE_EROSION);
        else {
          const st = strainOf(tag);
          if (!st || Date.now() - (st.last || 0) > ABANDONED_MS) strength[c] -= OFFLINE_EROSION;
        }
        if (strength[c] <= 0) { owner[c] = ''; strength[c] = 0; dirty = true; }
        continue;
      }
      if (nat === 1) {           // drought: no growth, and the land dries out
        strength[c] = Math.max(0.05, strength[c] - 0.012 * (T.drought ?? 1));
        continue;
      }
      let grow = 0.03 * p, spreadP = (0.15 * p + 0.03) * T.s;
      if (nat === 2) { grow *= 2 * (T.bloom || 1); spreadP *= 1.6; }
      strength[c] = Math.min(1, strength[c] + grow);
      if (strength[c] < 0.4 || Math.random() > spreadP) continue;
      if (T.jump && Math.random() < T.jump) {
        const j = idx(c % WORLD_W + Math.round((Math.random() - 0.5) * 9), ((c / WORLD_W) | 0) + Math.round((Math.random() - 0.5) * 9));
        if (!owner[j]) { claim(j, tag, 0.3); continue; }
      }
      const n = neighbours(c)[(Math.random() * 4) | 0];
      if (owner[n] === tag) continue;
      if (!owner[n]) { claim(n, tag, 0.3); continue; }
      const victim = owner[n], D = traitOf(victim);
      let att = strength[c] * (0.3 + p) * T.a, def = strength[n] * (0.3 + pw(victim)) * D.d;
      if (T.scav && pw(victim) === 0) att *= 2;
      if (D.colony) def *= 1 + 0.15 * neighbours(n).filter(k => owner[k] === victim).length;
      if (Math.random() < 0.25 * att / (att + def)) {
        const prev = claim(n, tag, 0.25);
        if (prev) overrun.set(tag + '>' + prev, (overrun.get(tag + '>' + prev) || 0) + 1);
      } else {
        strength[n] = Math.max(0.05, strength[n] - 0.02 * att);
        if (T.leech) strength[c] = Math.min(1, strength[c] + 0.01 * att);
      }
    }
    for (const [k, n] of overrun) if (n >= 3) {
      const [a, b] = k.split('>');
      note(b === BLIGHT ? `${a} burned out ${n} cells of blight` : `${a} overran ${n} cells of ${b}`);
    }
    for (const [t, n] of infected) if (n >= 3) note(`the blight took ${n} cells from ${t}`);
    infected.clear();
  }
  const timer = setInterval(tick, TICK_MS); timer.unref();
  const saver = setInterval(save, 30_000); saver.unref();

  function state() {
    const counts = new Map();
    for (const t of owner) if (t) counts.set(t, (counts.get(t) || 0) + 1);
    for (const [t, n] of counts) if (t !== BLIGHT && n > (peak.get(t) || 0)) { peak.set(t, n); dirty = true; }
    const palette = [...counts.keys()];
    const pi = new Map(palette.map((t, i) => [t, i + 1]));
    const own = new Uint8Array(CELLS), str = new Uint8Array(CELLS);
    for (let c = 0; c < CELLS; c++) { own[c] = owner[c] ? pi.get(owner[c]) : 0; str[c] = Math.round(strength[c] * 255); }
    const targets = blight.aff.map((v, s) => [v, s]).slice(1).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, s]) => s);
    return {
      W: WORLD_W, tick: tickNo, now: Date.now(), born,
      palette: palette.map(t => {
        const st = t === BLIGHT ? null : strainOf(t);
        return { tag: t, cells: counts.get(t), peak: peak.get(t) || counts.get(t), power: Math.round(power.get(t) || 0),
          node: st?.node || null, first: st?.first || null, last: st?.last || null, work: st?.work || 0 };
      }),
      owner: Buffer.from(own).toString('base64'), strength: Buffer.from(str).toString('base64'),
      events: events.slice(0, 8),
      nature: nature.map(e => ({ ...e })),
      blight: { tag: BLIGHT, gen: blight.gen, vir: +blight.vir.toFixed(3), spread: +blight.spread.toFixed(3), targets, cells: counts.get(BLIGHT) || 0 },
      network: { ...network(), history: powerHistory.slice(-60) },
    };
  }

  return { load, save, observe, state, tick, WORLD_W, _debug: { setNature: n => { nature = n; }, blight: () => blight } };
}
