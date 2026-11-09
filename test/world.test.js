import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createWorld, WORLD_W, BLIGHT } from '../world.js';
import { strainSignature, SIGNATURES } from '../public/field/genebank.js';

const strains = { aaaaaaaa: { node: { x: 0.25, y: 0.25 } }, bbbbbbbb: { node: { x: 0.75, y: 0.75 } } };
const make = async () => createWorld({ dataDir: await mkdtemp(join(tmpdir(), 'world-')), strainOf: t => strains[t] || null });
const held = (w, tag) => {
  const s = w.state(), own = Buffer.from(s.owner, 'base64'), i = s.palette.findIndex(p => p.tag === tag) + 1;
  return i ? own.filter(v => v === i).length : 0;
};
const report = (w, sid, strain, rate, obs = []) => w.observe({ sid, strain, rate, obs }, '1.2.3.4');
let sidN = 0;
// a viewer watching for `ticks` seconds: reports its GPU rate every 5 ticks, like the client does
const watch = (w, strain, ticks, rate = 12e6) => { for (let t = 0; t < ticks; t++) { if (t % 5 === 0) report(w, 'watch' + (sidN++) + 'xxxx', strain, rate); w.tick(); } };

test('a watched strain spores at its node and permeates outward', async () => {
  const w = await make();
  assert.equal(report(w, 'sessiona1', 'aaaaaaaa', 12e6).status, 200);
  w.tick();
  assert.ok(held(w, 'aaaaaaaa') >= 1, 'spore lands on the node');
  watch(w, 'aaaaaaaa', 200);
  assert.ok(held(w, 'aaaaaaaa') > 20, `grew to ${held(w, 'aaaaaaaa')} cells`);
});

test('rejects bad reports, floods, and unknown strains', async () => {
  const w = await make();
  assert.equal(w.observe({ sid: 'x', strain: 'aaaaaaaa' }).status, 400);
  assert.equal(report(w, 'sessionb1', 'aaaaaaaa', 1e6).status, 200);
  assert.equal(report(w, 'sessionb1', 'aaaaaaaa', 1e6).status, 429, 'per-session interval');
  report(w, 'sessionb2', 'cccccccc', 12e6, [[0, 'cccccccc', 64]]);   // unregistered strain: no evidence counted
  w.tick();
  assert.equal(held(w, 'cccccccc'), 0);
});

test('dish evidence only claims along a strain\'s own front', async () => {
  const w = await make();
  watch(w, 'aaaaaaaa', 60);
  const far = Math.floor(0.75 * WORLD_W) * WORLD_W + Math.floor(0.75 * WORLD_W);   // nowhere near a's territory
  for (let r = 0; r < 5; r++) { report(w, 'farclaim' + r + 'xx', 'aaaaaaaa', 12e6, [[far, 'aaaaaaaa', 64]]); w.tick(); }
  const s = w.state(), own = Buffer.from(s.owner, 'base64');
  assert.equal(own[far], 0, 'a far cell cannot be claimed by reports alone');
});

test('a strain whose viewers leave erodes while a watched one keeps growing', async () => {
  const w = await make();
  for (let t = 0; t < 120; t++) {
    if (t % 5 === 0) { report(w, 'both' + t + 'aaaa', 'aaaaaaaa', 12e6); report(w, 'both' + t + 'bbbb', 'bbbbbbbb', 12e6); }
    w.tick();
  }
  const a0 = held(w, 'aaaaaaaa'), b0 = held(w, 'bbbbbbbb');
  assert.ok(a0 > 10 && b0 > 10, `both grew (${a0}, ${b0})`);
  watch(w, 'aaaaaaaa', 400);           // b's viewer is gone
  assert.ok(held(w, 'bbbbbbbb') < b0, `b eroded from ${b0} to ${held(w, 'bbbbbbbb')}`);
  assert.ok(held(w, 'aaaaaaaa') > a0, `a kept growing from ${a0} to ${held(w, 'aaaaaaaa')}`);
});

test('reports the live network total from every recent viewer', async () => {
  const w = await make();
  report(w, 'viewerone1', 'aaaaaaaa', 3e6);
  report(w, 'viewertwo2', 'bbbbbbbb', 5e6);
  report(w, 'viewerhid3', 'bbbbbbbb', 0);        // a hidden tab reports zero and isn't counted as contributing
  w.tick();
  assert.deepEqual({ total: w.state().network.total, viewers: w.state().network.viewers }, { total: 8e6, viewers: 2 });
  for (let t = 0; t < 20; t++) w.tick();         // nobody reports again
  assert.equal(w.state().network.viewers, 0);
});

test('unwatched territory goes dormant instead of vanishing, and the world remembers its age', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'world-'));
  const recent = { dddddddd: { node: { x: 0.5, y: 0.5 }, first: Date.now() - 3 * 86400_000, last: Date.now() - 3600_000 } };
  const w = createWorld({ dataDir: dir, strainOf: t => recent[t] || null });
  watch(w, 'dddddddd', 150);
  for (let t = 0; t < 120; t++) w.tick();       // the last viewer's power fades out
  const grown = held(w, 'dddddddd');
  assert.ok(grown > 10, `grew to ${grown}`);
  for (let t = 0; t < 1500; t++) w.tick();      // nobody watches for 25 minutes
  assert.equal(held(w, 'dddddddd'), grown, 'dormant territory stays put');
  const s = w.state(), str = Buffer.from(s.strength, 'base64'), own = Buffer.from(s.owner, 'base64');
  const di = s.palette.findIndex(p => p.tag === 'dddddddd') + 1;
  assert.ok(str.every((v, c) => own[c] !== di || v <= 0.13 * 255), 'but it weakens to the dormant floor');
  assert.ok(s.palette[di - 1].peak >= grown);
  await w.save();
  const again = createWorld({ dataDir: dir, strainOf: t => recent[t] || null });
  await again.load();
  assert.equal(again.state().born, s.born);
  assert.equal(held(again, 'dddddddd'), grown);
});

// a strain tag whose signature gene is the given effect name
const tagWith = (noun, skip = 0) => { for (let i = 1; ; i++) { const t = (i * 2654435761) >>> 0; if (SIGNATURES[strainSignature(t)][0] === noun && skip-- <= 0) return t.toString(16).padStart(8, '0'); } };

test('signature traits shape territory: darters outspread holdfasts', async () => {
  const fast = tagWith('Darters'), slow = tagWith('Holdfast');
  let f = 0, s = 0;
  for (let r = 0; r < 4; r++) {
    const reg = { [fast]: { node: { x: 0.25, y: 0.25 }, last: Date.now() }, [slow]: { node: { x: 0.75, y: 0.75 }, last: Date.now() } };
    const w = createWorld({ dataDir: await mkdtemp(join(tmpdir(), 'world-')), strainOf: t => reg[t] || null });
    for (let t = 0; t < 150; t++) { if (t % 5 === 0) { report(w, 'tf' + r + t + 'xxxx', fast, 12e6); report(w, 'ts' + r + t + 'xxxx', slow, 12e6); } w.tick(); }
    f += held(w, fast); s += held(w, slow);
  }
  assert.ok(f > s * 1.3, `darters ${f} vs holdfast ${s}`);
});

test('nature: a drought over a strain land sets it back', async () => {
  const w = await make();
  watch(w, 'aaaaaaaa', 120);
  const str = () => { const s = w.state(), o = Buffer.from(s.owner, 'base64'), v = Buffer.from(s.strength, 'base64'); let t = 0; o.forEach((k, c) => { if (k) t += v[c]; }); return t; };
  const before = str();
  w._debug.setNature([{ kind: 'drought', x: 16, y: 16, r: 14, left: 200 }]);
  watch(w, 'aaaaaaaa', 40);
  assert.ok(str() < before, `strength fell from ${before} to ${str()}`);
});

test('chaos: the blight breaks out, takes land, and mutates its own genes', async () => {
  const w = await make();
  watch(w, 'aaaaaaaa', 200);
  const gen0 = w._debug.blight().gen, vir0 = w._debug.blight().vir;
  let ever = 0;
  for (let t = 0; t < 1500; t++) { if (t % 5 === 0) report(w, 'chaos' + t + 'xx', 'aaaaaaaa', 12e6); w.tick(); ever = Math.max(ever, w.state().blight.cells); }
  assert.ok(ever > 0, 'an outbreak happened');
  assert.ok(w._debug.blight().gen > gen0 || w._debug.blight().vir !== vir0, 'the blight evolved');
  assert.equal(w.observe({ sid: 'fakeblight1', strain: BLIGHT, rate: 1 }).status, 400, 'nobody can report as the blight');
});
