import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createSharedBank, SHARED_F, SHARED_HMAX } from '../shared-bank.js';
import { layout, makeFounder, strainTagOf, genomeToB64 } from '../public/field/genebank.js';

const L = layout(SHARED_F, SHARED_HMAX);
const base = { F: SHARED_F, HMAX: SHARED_HMAX, GS: L.GS };
const hex = tag => tag.toString(16).padStart(8, '0');

test('accepts valid genomes, rejects malformed ones, and rate-limits per address', async () => {
  const bank = createSharedBank({ dataDir: await mkdtemp(join(tmpdir(), 'bank-')) });
  const tag = strainTagOf('visitor-a');
  const good = makeFounder(L, tag);
  const bad = makeFounder(L, tag); bad[0] = 99;                 // impossible hidden width
  const out = bank.accept({ ...base, sid: 'sessionaaaa', strain: hex(tag), work: 1000,
    entries: [{ g: genomeToB64(good), fitness: 3, kids: 3 }, { g: genomeToB64(bad), fitness: 3, kids: 3 }, { g: 'nope', kids: 1 }] }, '1.1.1.1');
  assert.equal(out.status, 200);
  assert.equal(out.body.accepted, 1);
  assert.equal(out.body.rejected, 2);
  assert.equal(bank.accept({ ...base, entries: [] }, '1.1.1.1').status, 429);
  assert.equal(bank.accept({ F: 4, HMAX: 12, GS: 1, entries: [] }, '2.2.2.2').status, 400);
});

test('credits capped GPU work to the strain and lists it with its niches', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'bank-'));
  const bank = createSharedBank({ dataDir: dir });
  const tag = strainTagOf('visitor-b');
  const g7 = makeFounder(L, tag); g7[5] = 7;
  bank.accept({ ...base, sid: 'sessionbbbb', strain: hex(tag), work: 1e18, node: { x: 0.25, y: 0.75, chosen: true },
    entries: [{ g: genomeToB64(g7), fitness: 5, kids: 5 }] }, '3.3.3.3');
  const listed = bank.list(0, 'sessionbbbb', hex(tag));
  assert.equal(listed.entries.length, 1);
  assert.equal(listed.mine.tag, hex(tag));
  assert.equal(listed.mine.niches, 1);
  assert.equal(listed.mine.bestGen, 7, "first upload records the strain's generation");
  assert.ok(listed.mine.work > 0 && listed.mine.work <= 300 * 262144 * 60, 'work is capped by elapsed time');
  assert.equal(listed.online, 1);
  assert.deepEqual(listed.mine.node, { x: 0.25, y: 0.75, chosen: true });
  await bank.save();
  const saved = JSON.parse(await readFile(join(dir, 'bank.json'), 'utf8'));
  assert.equal(saved.entries.length, 1);
  assert.ok(saved.strains[hex(tag)].work > 0);
  const reloaded = createSharedBank({ dataDir: dir });
  await reloaded.load();
  assert.equal(reloaded.list(0).niches, 1);
});
