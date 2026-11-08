// Every creature owns these weights. No backpropagation or per-action API calls.
export const INPUTS = 32, HIDDEN_A = 64, HIDDEN_B = 96;
export const ACTIONS = [
  'north', 'east', 'south', 'west',
  'dig north', 'dig east', 'dig south', 'dig west',
  'drop sand', 'harvest', 'share', 'food signal', 'home signal',
  'reproduce', 'rest', 'explore'
];
export const PARAMETERS = INPUTS * HIDDEN_A + HIDDEN_A * HIDDEN_B + HIDDEN_B * ACTIONS.length
  + HIDDEN_A + HIDDEN_B + ACTIONS.length;

const randomWeight = fanIn => (Math.random() * 2 - 1) * 1.4 / Math.sqrt(fanIn);

export function createGenome(parent = null, mutation = null) {
  const weights = new Float32Array(PARAMETERS);
  if (parent) {
    weights.set(parent);
    const rate = mutation?.rate ?? .018;
    let changes = 0, magnitude = 0;
    for (let i = 0; i < weights.length; i++) {
      if (Math.random() < rate) {
        const delta = (Math.random() * 2 - 1) * .24;
        weights[i] += delta; changes++; magnitude += Math.abs(delta);
      }
    }
    // Rare block mutations create clearly divergent descendants.
    if (mutation && Math.random() < .16) {
      const start = Math.floor(Math.random() * (weights.length - 96));
      for (let i = start; i < start + 96; i++) {
        const delta = (Math.random() * 2 - 1) * .5;
        weights[i] += delta; magnitude += Math.abs(delta);
      }
      mutation.macro = true; changes += 96;
    }
    if (mutation) { mutation.neuralWeights = changes; mutation.neuralMagnitude = +magnitude.toFixed(2); }
  } else {
    let offset = 0;
    for (const [count, fanIn] of [[INPUTS * HIDDEN_A, INPUTS], [HIDDEN_A * HIDDEN_B, HIDDEN_A],
      [HIDDEN_B * ACTIONS.length, HIDDEN_B]]) {
      for (let i = 0; i < count; i++) weights[offset++] = randomWeight(fanIn);
    }
    offset += HIDDEN_A + HIDDEN_B;
    // A broad starting action distribution; the sensor-to-action weights remain random.
    for (let i = 0; i < ACTIONS.length; i++) weights[offset + i] = i < 4 ? .45 : 0;
  }
  return weights;
}

export function createController(genome = createGenome()) {
  return { genome, a: new Float32Array(HIDDEN_A), b: new Float32Array(HIDDEN_B),
    outputs: new Float32Array(ACTIONS.length), ticks: 0 };
}

export function chooseAction(controller, senses, random = Math.random, modifiers = null) {
  const { genome, a, b, outputs } = controller;
  let offset = 0;
  for (let h = 0; h < HIDDEN_A; h++) {
    let sum = 0;
    for (let i = 0; i < INPUTS; i++) sum += genome[offset++] * (senses[i] || 0);
    a[h] = Math.tanh(sum + genome[INPUTS * HIDDEN_A + HIDDEN_A * HIDDEN_B
      + HIDDEN_B * ACTIONS.length + h] + a[h] * .28);
  }
  const secondBias = INPUTS * HIDDEN_A + HIDDEN_A * HIDDEN_B + HIDDEN_B * ACTIONS.length + HIDDEN_A;
  for (let h = 0; h < HIDDEN_B; h++) {
    let sum = 0;
    for (let i = 0; i < HIDDEN_A; i++) sum += genome[offset++] * a[i];
    b[h] = Math.tanh(sum + genome[secondBias + h] + b[h] * .22);
  }
  const outputBias = secondBias + HIDDEN_B;
  let total = 0;
  for (let action = 0; action < ACTIONS.length; action++) {
    let sum = genome[outputBias + action];
    for (let i = 0; i < HIDDEN_B; i++) sum += genome[offset++] * b[i];
    outputs[action] = Math.exp(Math.max(-12, Math.min(12, sum))) * (modifiers?.[action] ?? 1);
    total += outputs[action];
  }
  controller.ticks++;
  let draw = random() * total;
  for (let i = 0; i < outputs.length; i++) if ((draw -= outputs[i]) <= 0) return i;
  return outputs.length - 1;
}
