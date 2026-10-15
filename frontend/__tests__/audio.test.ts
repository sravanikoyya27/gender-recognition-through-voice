/**
 * Unit tests for the pure audio helpers, run with Node's built-in test runner (`npm test`).
 * No browser APIs are needed: trimming and WAV encoding are plain array/DataView code.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { encodeWav, formatTime, trimToMaxDuration, wavDurationSeconds } from "../lib/audio.ts";

const RATE = 16_000;
const MAX_SECONDS = 10;

function samplesOf(seconds: number): Float32Array {
  return new Float32Array(Math.round(seconds * RATE)).fill(0.25);
}

describe("trimToMaxDuration", () => {
  it("leaves a take shorter than the maximum untouched (same array)", () => {
    const input = samplesOf(2.5);
    const output = trimToMaxDuration(input, RATE, MAX_SECONDS);
    assert.equal(output, input);
    assert.equal(output.length, 2.5 * RATE);
  });

  it("leaves a take of exactly the maximum untouched", () => {
    const input = samplesOf(MAX_SECONDS);
    const output = trimToMaxDuration(input, RATE, MAX_SECONDS);
    assert.equal(output.length, MAX_SECONDS * RATE);
    assert.equal(output, input);
  });

  it("cuts a take that is one sample over the maximum", () => {
    const input = new Float32Array(MAX_SECONDS * RATE + 1);
    const output = trimToMaxDuration(input, RATE, MAX_SECONDS);
    assert.equal(output.length, MAX_SECONDS * RATE);
  });

  it("cuts a take that ran a few hundred milliseconds over the auto-stop (10.07 s -> 10.00 s)", () => {
    const input = samplesOf(10.07);
    const output = trimToMaxDuration(input, RATE, MAX_SECONDS);
    assert.equal(output.length / RATE, MAX_SECONDS);
    // The kept part is the beginning of the recording, not the end.
    assert.equal(output[0], input[0]);
  });

  it("supports fractional limits and ignores invalid ones", () => {
    assert.equal(trimToMaxDuration(samplesOf(3), RATE, 1.5).length, Math.floor(1.5 * RATE));
    assert.equal(trimToMaxDuration(samplesOf(3), RATE, undefined).length, 3 * RATE);
    assert.equal(trimToMaxDuration(samplesOf(3), RATE, 0).length, 3 * RATE);
    assert.equal(trimToMaxDuration(samplesOf(3), RATE, Number.NaN).length, 3 * RATE);
  });
});

describe("encodeWav", () => {
  it("writes a 16 kHz mono 16-bit PCM header whose data length matches the samples", () => {
    const wav = encodeWav(samplesOf(1), RATE);
    assert.equal(wav.type, "audio/wav");
    assert.equal(wav.size, 44 + RATE * 2);
    assert.equal(wavDurationSeconds(wav, RATE), 1);
  });

  it("produces a WAV whose duration never exceeds the maximum after trimming", () => {
    const trimmed = trimToMaxDuration(samplesOf(10.07), RATE, MAX_SECONDS);
    const wav = encodeWav(trimmed, RATE);
    assert.ok(wavDurationSeconds(wav, RATE) <= MAX_SECONDS);
    assert.equal(wavDurationSeconds(wav, RATE), MAX_SECONDS);
  });

  it("clips out-of-range samples instead of wrapping", () => {
    const wav = encodeWav(new Float32Array([2, -2, 0]), RATE);
    assert.equal(wav.size, 44 + 3 * 2);
  });
});

describe("formatTime", () => {
  it("formats minutes, seconds and optional tenths", () => {
    assert.equal(formatTime(0), "00:00");
    assert.equal(formatTime(9.96), "00:09");
    assert.equal(formatTime(9.96, true), "00:09.9");
    assert.equal(formatTime(65.2, true), "01:05.2");
    assert.equal(formatTime(-3), "00:00");
  });
});
