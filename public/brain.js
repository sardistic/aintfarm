export const TASKS = ['digger', 'forager', 'nurse', 'rest'];
const INPUTS = 8, HIDDEN = 8;
const jitter = amount => (Math.random() * 2 - 1) * amount;
const clamp = (n, low, high) => Math.max(low, Math.min(high, n));

// A small, individually varied recurrent controller. Its weights are initialized
// around a useful sensory circuit, then its activations change through experience.
export function createBrain(preferred = 'digger') {
  const inputWeights = new Float32Array(HIDDEN * INPUTS);
  const outputWeights = new Float32Array(TASKS.length * HIDDEN);
  for (let h = 0; h < HIDDEN; h++) for (let i = 0; i < INPUTS; i++)
    inputWeights[h * INPUTS + i] = (h === i ? 1.9 : 0) + jitter(.12);
  const base = [
    [1.7, 0, 0, .5, -.6, 0, 0, .6],
    [0, 1.6, 0, 0, -.5, .5, 0, 0],
    [0, 0, 1.5, 0, -.3, 0, .45, 0],
    [0, 0, 0, 0, 2, 0, 0, 0]
  ];
  for (let task = 0; task < TASKS.length; task++) for (let h = 0; h < HIDDEN; h++)
    outputWeights[task * HIDDEN + h] = base[task][h] + jitter(.12);
  const bias = new Float32Array([.08, .17, .06, -.75]);
  bias[TASKS.indexOf(preferred)] += .27;
  return { inputWeights, outputWeights, bias, hidden: new Float32Array(HIDDEN),
    outputs: new Float32Array(TASKS.length), senses: new Float32Array(INPUTS), ticks: 0 };
}

export function evaluateBrain(brain, senses, currentTask) {
  for (let i = 0; i < INPUTS; i++) brain.senses[i] = clamp(senses[i] ?? 0, 0, 1);
  for (let h = 0; h < HIDDEN; h++) {
    let signal = brain.hidden[h] * .28;
    for (let i = 0; i < INPUTS; i++) signal += brain.inputWeights[h * INPUTS + i] * brain.senses[i];
    brain.hidden[h] += (Math.tanh(signal) - brain.hidden[h]) * .38;
  }
  let bestTask = currentTask, bestValue = -Infinity;
  for (let task = 0; task < TASKS.length; task++) {
    let value = brain.bias[task] + (TASKS[task] === currentTask ? .16 : 0);
    for (let h = 0; h < HIDDEN; h++) value += brain.outputWeights[task * HIDDEN + h] * brain.hidden[h];
    brain.outputs[task] = value;
    if (value > bestValue) { bestValue = value; bestTask = TASKS[task]; }
  }
  brain.ticks++;
  return bestTask;
}

export function rewardBrain(brain, taskName, reward = 1) {
  const task = TASKS.indexOf(taskName);
  if (task < 0) return;
  brain.bias[task] = clamp(brain.bias[task] + reward * .012, -1, 1.8);
  for (let h = 0; h < HIDDEN; h++) {
    const i = task * HIDDEN + h;
    brain.outputWeights[i] = clamp(brain.outputWeights[i] + reward * brain.hidden[h] * .004, -2.5, 2.5);
  }
}
