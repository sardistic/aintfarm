import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createWorld, WORLD_W } from '../world.js';

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
  assert.equal(held(w, 'aaaaaaaa'), 1, 'spore lands on the node');
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
