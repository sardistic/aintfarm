// The shared world: one territory map everyone sees. Strains grow from their nodes like lichen, powered by the GPUs
// of viewers currently watching, and borders move according to which strains actually survive in viewers' dishes.
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

export const WORLD_W = 64;                  // territory cells per side (each covers 12×12 dish cells at G=768)
const CELLS = WORLD_W * WORLD_W;
const HEX8 = /^[0-9a-f]{8}$/;
const MAX_RATE = 12e6;                      // particle-steps/s; GPU power saturates here so one card can't dominate
const OBSERVE_INTERVAL_MS = 4000;           // per session
const POWER_DECAY = 0.85;                   // per tick, once a strain has stopped reporting
const REPORT_GRACE_TICKS = 12;              // clients report every 5 s; allow a missed report before fading
const OFFLINE_EROSION = 0.004;              // strength lost per tick by cells of strains with no power
const TICK_MS = 1000;

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
  const file = join(dataDir, 'world.json');
  let tickNo = 0, dirty = false;

  const idx = (x, y) => ((y + WORLD_W) % WORLD_W) * WORLD_W + ((x + WORLD_W) % WORLD_W);
  const neighbours = c => { const x = c % WORLD_W, y = (c / WORLD_W) | 0; return [idx(x + 1, y), idx(x - 1, y), idx(x, y + 1), idx(x, y - 1)]; };
  const pw = tag => Math.sqrt(Math.min(power.get(tag) || 0, MAX_RATE) / MAX_RATE);   // 0..1
  function note(text) { events.unshift({ t: Date.now(), text }); events.length = Math.min(events.length, 20); }

  async function load() {
    try {
      const o = JSON.parse(await readFile(file, 'utf8'));
      if (o.W !== WORLD_W) return;
      o.owner.forEach((t, i) => { if (HEX8.test(t)) owner[i] = t; });
      o.strength.forEach((s, i) => { strength[i] = Math.max(0, Math.min(1, +s || 0)); });
      console.log(`[world] loaded territory (${owner.filter(Boolean).length} cells held)`);
    } catch (e) { if (e.code !== 'ENOENT') console.warn('[world] could not load:', e.message); }
  }
  async function save() {
    if (!dirty) return;
    dirty = false;
    try {
      await mkdir(dataDir, { recursive: true });
      await writeFile(file + '.tmp', JSON.stringify({ W: WORLD_W, owner, strength: Array.from(strength, s => +s.toFixed(3)) }));
      await rename(file + '.tmp', file);
    } catch (e) { dirty = true; console.warn('[world] could not save:', e.message); }
  }

  // A viewer's dish reports which strains it saw surviving where (sampled particles), plus its GPU throughput.
  function observe(body, ip) {
    const now = Date.now();
    const sid = typeof body?.sid === 'string' && /^[a-z0-9]{8,32}$/.test(body.sid) ? body.sid : null;
    if (!sid || !HEX8.test(body.strain || '')) return { status: 400, body: { error: 'Bad request.' } };
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

  function claim(c, tag, s, why) {
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

  function tick() {
    tickNo++;
    if (tickNo % 5 === 0) { powerHistory.push(Math.round(network().total)); if (powerHistory.length > 120) powerHistory.shift(); }
    const overrun = new Map();   // "a>b" -> cells taken this tick
    for (const [tag, p] of power) {
      if (tickNo - (lastReportTick.get(tag) ?? -1e9) > REPORT_GRACE_TICKS) power.set(tag, p * POWER_DECAY);
      if (power.get(tag) < 1000) { power.delete(tag); lastReportTick.delete(tag); }
    }
    reported.clear();
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
      // 2) lichen growth: established cells of powered strains creep into their neighbours
      const tag = owner[c];
      if (!tag) continue;
      const p = pw(tag);
      if (p === 0) {
        strength[c] -= OFFLINE_EROSION;
        if (strength[c] <= 0) { owner[c] = ''; strength[c] = 0; dirty = true; }
        continue;
      }
      strength[c] = Math.min(1, strength[c] + 0.03 * p);
      if (strength[c] < 0.4 || Math.random() > 0.15 * p + 0.03) continue;
      const n = neighbours(c)[(Math.random() * 4) | 0];
      if (owner[n] === tag) continue;
      if (!owner[n]) { claim(n, tag, 0.3); continue; }
      const att = strength[c] * (0.3 + p), def = strength[n] * (0.3 + pw(owner[n]));
      if (Math.random() < 0.25 * att / (att + def)) {
        const prev = claim(n, tag, 0.25);
        if (prev) overrun.set(tag + '>' + prev, (overrun.get(tag + '>' + prev) || 0) + 1);
      } else strength[n] = Math.max(0.05, strength[n] - 0.02 * att);
    }
    for (const [k, n] of overrun) if (n >= 3) { const [a, b] = k.split('>'); note(`${a} overran ${n} cells of ${b}`); }
  }
  const timer = setInterval(tick, TICK_MS); timer.unref();
  const saver = setInterval(save, 30_000); saver.unref();

  function state() {
    const counts = new Map();
    for (const t of owner) if (t) counts.set(t, (counts.get(t) || 0) + 1);
    const palette = [...counts.keys()];
    const pi = new Map(palette.map((t, i) => [t, i + 1]));
    const own = new Uint8Array(CELLS), str = new Uint8Array(CELLS);
    for (let c = 0; c < CELLS; c++) { own[c] = owner[c] ? pi.get(owner[c]) : 0; str[c] = Math.round(strength[c] * 255); }
    return {
      W: WORLD_W, tick: tickNo, now: Date.now(),
      palette: palette.map(t => ({ tag: t, cells: counts.get(t), power: Math.round(power.get(t) || 0), node: strainOf(t)?.node || null })),
      owner: Buffer.from(own).toString('base64'), strength: Buffer.from(str).toString('base64'),
      events: events.slice(0, 8),
      network: { ...network(), history: powerHistory.slice(-60) },
    };
  }

  return { load, save, observe, state, tick, WORLD_W };
}
