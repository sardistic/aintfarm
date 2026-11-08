export const TRAIT_NAMES = ['mobility', 'digging', 'foraging', 'sociability',
  'aggression', 'secretion', 'sensing', 'size', 'metabolism', 'fecundity', 'longevity', 'mutability'];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const rand = (low, high) => low + Math.random() * (high - low);

export function makePhenotype(parent = null) {
  const traits = {}, changes = [];
  for (const name of TRAIT_NAMES) {
    if (!parent) { traits[name] = rand(.2, .8); continue; }
    const before = parent.traits[name];
    const burst = Math.random() < .12;
    const delta = burst ? rand(-.48, .48) : Math.random() < .62 ? rand(-.16, .16) : 0;
    traits[name] = clamp(before + delta, .02, .98);
    if (Math.abs(traits[name] - before) >= .015)
      changes.push({ name, before: +before.toFixed(2), after: +traits[name].toFixed(2),
        delta: +(traits[name] - before).toFixed(2), burst });
  }
  const signature = new Float32Array(4), receptor = new Float32Array(4);
  for (let channel = 0; channel < 4; channel++) {
    for (const [label, target] of [['emission', signature], ['reception', receptor]]) {
      const before = parent ? parent[label === 'emission' ? 'signature' : 'receptor'][channel] : rand(-1, 1);
      const delta = parent ? (Math.random() < .16 ? rand(-.9, .9) : rand(-.22, .22)) : 0;
      target[channel] = clamp(before + delta, -1, 1);
      if (parent && Math.abs(target[channel] - before) >= .05)
        changes.push({ name: `${label} ${channel + 1}`, before: +before.toFixed(2),
          after: +target[channel].toFixed(2), delta: +(target[channel] - before).toFixed(2),
          burst: Math.abs(delta) > .4 });
    }
  }
  const hue = parent ? ((parent.color + rand(-16, 16) + (Math.random() < .14 ? rand(-85, 85) : 0)) % 360 + 360) % 360
    : rand(0, 360);
  const hueDelta = parent ? ((hue - parent.color + 540) % 360) - 180 : 0;
  if (parent && Math.abs(hueDelta) > 5)
    changes.push({ name: 'pigment', before: Math.round(parent.color), after: Math.round(hue),
      delta: Math.round(hueDelta), burst: Math.abs(hueDelta) > 35 });
  if (parent && !changes.length) {
    const before = traits.mobility;
    traits.mobility = clamp(before + (Math.random() < .5 ? -.06 : .06), .02, .98);
    changes.push({ name: 'mobility', before: +before.toFixed(2),
      after: +traits.mobility.toFixed(2), delta: +(traits.mobility - before).toFixed(2), burst: false });
  }
  const magnitude = changes.reduce((sum, item) => sum + Math.abs(item.delta), 0);
  return { traits, signature, receptor, hue, changes,
    mutation: { traitChanges: changes.length, bursts: changes.filter(item => item.burst).length,
      magnitude: +magnitude.toFixed(2), changes } };
}
