// The shared gene bank: one niche archive that every visitor's Field Life contributes to and draws immigrants from.
// Uses the same genome layout/embedding as the browser (public/field/genebank.js).
import { readFile, writeFile, rename, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { layout, makeGenomeTools, b64ToGenome, genomeToB64 } from './public/field/genebank.js';

export const SHARED_F = 8, SHARED_HMAX = 12;          // only the default configuration shares a bank
const L = layout(SHARED_F, SHARED_HMAX);
const { validateGenome, GeneBank } = makeGenomeTools(L);

const MAX_PER_POST = 16;
const MAX_FITNESS = 64;                                 // claims can't be verified, so they are capped...
const HALF_LIFE_MS = 6 * 3600 * 1000;                   // ...and decay, so stale or inflated entries get displaced
const POST_INTERVAL_MS = 20_000;                        // per session
const IP_POSTS_PER_MIN = 12;                            // per client address (households and offices share one)
const GLOBAL_INSERTS_PER_MIN = 600;
const ONLINE_WINDOW_MS = 90_000;
const BODY_LIMIT = 96 * 1024;
const MAX_WORK_RATE = 300 * 262144;                     // particle-steps per second one browser could plausibly do
const HEX8 = /^[0-9a-f]{8}$/;

export function createSharedBank({ dataDir }) {
  const bank = new GeneBank({ cap: 400, halfLifeMs: HALF_LIFE_MS, fastPrune: true });
  const file = join(dataDir, 'bank.json');
  const lastPost = new Map();      // session (or ip) -> ms
  const ipPosts = new Map();       // ip -> [window start, count]
  const seen = new Map();          // session id -> ms
  let dirty = false, insertsThisMinute = 0, minuteStart = Date.now(), contributors = new Set();
  const strains = new Map();       // strain tag (hex) -> { work, bestGen, first, last, visitors }
  const lastWork = new Map();      // session id -> ms of last credited work

  async function load() {
    try {
      const o = JSON.parse(await readFile(file, 'utf8'));
      bank.load(o);
      bank.inserts = o.inserts || 0;
      contributors = new Set(o.contributors || []);
      for (const [tag, st] of Object.entries(o.strains || {})) if (HEX8.test(tag)) strains.set(tag, { work: +st.work || 0, bestGen: +st.bestGen || 0, first: +st.first || Date.now(), last: +st.last || 0, visitors: +st.visitors || 1, node: st.node || null });
      console.log(`[bank] loaded ${bank.entries.length} niches from ${file}`);
    } catch (e) { if (e.code !== 'ENOENT') console.warn('[bank] could not load:', e.message); }
  }
  async function save() {
    if (!dirty) return;
    dirty = false;
    try {
      await mkdir(dataDir, { recursive: true });
      const tmp = `${file}.tmp`;
      await writeFile(tmp, JSON.stringify({ ...bank.toJSON(), inserts: bank.inserts, contributors: [...contributors].slice(-5000), strains: Object.fromEntries(strains) }));
      await rename(tmp, file);
    } catch (e) { dirty = true; console.warn('[bank] could not save:', e.message); }
  }
  const timer = setInterval(save, 30_000);
  timer.unref();

  function online(sid) {
    const now = Date.now();
    if (typeof sid === 'string' && /^[a-z0-9]{8,32}$/.test(sid)) seen.set(sid, now);
    for (const [k, t] of seen) if (now - t > ONLINE_WINDOW_MS) seen.delete(k);
    return seen.size;
  }

  function strainTable(mineTag) {
    const niches = new Map();
    for (const e of bank.entries) { const t = (e.dec.lineage >>> 0).toString(16).padStart(8, '0'); niches.set(t, (niches.get(t) || 0) + 1); }
    const now = Date.now();
    const row = (tag, st) => ({ tag, work: Math.round(st.work), bestGen: st.bestGen, niches: niches.get(tag) || 0, online: now - st.last < ONLINE_WINDOW_MS, visitors: st.visitors, node: st.node || null });
    const top = [...strains].sort((a, b) => b[1].work - a[1].work).slice(0, 40).map(([t, st]) => row(t, st));
    const mine = HEX8.test(mineTag || '') && strains.has(mineTag) ? row(mineTag, strains.get(mineTag)) : null;
    return { strains: top, mine, strainCount: strains.size };
  }

  function list(since, sid, strain) {
    const entries = bank.entries.filter(e => e.t > since).map(e => ({ id: e.id, g: genomeToB64(e.genome), fitness: e.fitness, kids: e.kids, t: e.t }));
    return { F: SHARED_F, HMAX: SHARED_HMAX, GS: L.GS, now: Date.now(), niches: bank.entries.length, samples: bank.inserts,
      contributors: contributors.size, online: online(sid), ...strainTable(strain), entries };
  }

  function accept(body, ip) {
    const now = Date.now();
    const who = typeof body?.sid === 'string' && /^[a-z0-9]{8,32}$/.test(body.sid) ? 's:' + body.sid : 'i:' + ip;
    if (now - (lastPost.get(who) || 0) < POST_INTERVAL_MS) return { status: 429, body: { error: 'Too many uploads; try again shortly.' } };
    const win = ipPosts.get(ip);
    if (win && now - win[0] < 60_000 && win[1] >= IP_POSTS_PER_MIN) return { status: 429, body: { error: 'Too many uploads from this address.' } };
    if (!win || now - win[0] >= 60_000) ipPosts.set(ip, [now, 1]); else win[1]++;
    if (ipPosts.size > 10_000) for (const [k, w] of ipPosts) if (now - w[0] > 60_000) ipPosts.delete(k);
    if (now - minuteStart > 60_000) { minuteStart = now; insertsThisMinute = 0; }
    if (insertsThisMinute >= GLOBAL_INSERTS_PER_MIN) return { status: 429, body: { error: 'The shared bank is busy.' } };
    if (!body || body.F !== SHARED_F || body.HMAX !== SHARED_HMAX || body.GS !== L.GS || !Array.isArray(body.entries)) {
      return { status: 400, body: { error: 'Wrong configuration.' } };
    }
    lastPost.set(who, now);
    if (lastPost.size > 10_000) for (const [k, t] of lastPost) if (now - t > POST_INTERVAL_MS) lastPost.delete(k);
    let accepted = 0, rejected = 0, kept = 0;
    const good = [];
    for (const e of body.entries.slice(0, MAX_PER_POST)) {
      const g = typeof e?.g === 'string' && e.g.length < 4096 ? b64ToGenome(e.g) : null;
      const kids = Math.min(Math.max(0, Math.floor(Number(e?.kids) || 0)), MAX_FITNESS);
      const fitness = Math.min(Math.max(0, Number(e?.fitness) || 0), MAX_FITNESS);
      if (!g || validateGenome(g) || kids < 1) { rejected++; continue; }
      accepted++;
      insertsThisMinute++;
      if (bank.insert(g, fitness, kids, 'shared', now)) kept++;
      good.push(g);
    }
    const sidOk = typeof body.sid === 'string' && /^[a-z0-9]{8,32}$/.test(body.sid);
    if (sidOk) { contributors.add(body.sid); online(body.sid); }
    // credit this browser's GPU work to its strain, never more than it could have done since its last report
    if (sidOk && typeof body.strain === 'string' && HEX8.test(body.strain)) {
      const since = Math.min(120_000, Math.max(1000, now - (lastWork.get(body.sid) || now - 60_000)));
      const work = Math.min(Math.max(0, Number(body.work) || 0), MAX_WORK_RATE * since / 1000);
      lastWork.set(body.sid, now);
      if (lastWork.size > 20_000) for (const [k, t] of lastWork) if (now - t > 600_000) lastWork.delete(k);
      let st = strains.get(body.strain);
      if (!st) { st = { work: 0, bestGen: 0, first: now, last: now, visitors: 1 }; strains.set(body.strain, st); }
      if (now - st.last > 30 * 60_000) st.visitors++;
      const n = body.node;
      if (n && Number.isFinite(n.x) && Number.isFinite(n.y) && n.x >= 0 && n.x < 1 && n.y >= 0 && n.y < 1) {
        st.node = { x: Math.round(n.x * 1e4) / 1e4, y: Math.round(n.y * 1e4) / 1e4, chosen: !!n.chosen };
      }
      st.work += work; st.last = now;
      dirty = true;
    }
    // a strain's best generation (after the strain record exists, so a first upload counts too)
    for (const g of good) {
      const st = strains.get((g[4] >>> 0).toString(16).padStart(8, '0'));
      if (st && g[5] > st.bestGen) { st.bestGen = g[5]; dirty = true; }
    }
    if (kept) dirty = true;
    return { status: 200, body: { accepted, rejected, kept, niches: bank.entries.length } };
  }

  // for the shared world: a strain's record (node, work) and its champion genome (fittest archived net carrying its tag)
  const strainOf = tag => strains.get(tag) || null;
  // a strain becomes known to the world as soon as its viewer reports in (not only after a bank upload lands)
  function touch(tag, node) {
    if (!HEX8.test(tag || '')) return;
    const now = Date.now();
    let st = strains.get(tag);
    if (!st) { st = { work: 0, bestGen: 0, first: now, last: now, visitors: 1 }; strains.set(tag, st); dirty = true; }
    st.last = now;
    if (node && Number.isFinite(node.x) && Number.isFinite(node.y) && node.x >= 0 && node.x < 1 && node.y >= 0 && node.y < 1) {
      st.node = { x: Math.round(node.x * 1e4) / 1e4, y: Math.round(node.y * 1e4) / 1e4, chosen: !!node.chosen };
    }
  }
  function champions(tags) {
    const want = new Set(tags.filter(t => HEX8.test(t)).slice(0, 48)), best = new Map();
    for (const e of bank.entries) {
      const t = (e.dec.lineage >>> 0).toString(16).padStart(8, '0');
      if (want.has(t) && (!best.has(t) || e.fitness > best.get(t).fitness)) best.set(t, e);
    }
    return Object.fromEntries([...best].map(([t, e]) => [t, genomeToB64(e.genome)]));
  }

  return { load, save, list, accept, strainOf, touch, champions, BODY_LIMIT };
}
