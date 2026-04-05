import { test, expect } from 'bun:test';
import ImageClassifier from '../main';
import path from 'path';

test('ImageClassifier: classifies all real images as real (with high probability)', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();
  const results = await classifier.classifyAllImages();

  const correct = results.real.filter(r => r.correct).length;
  const total = results.real.length;
  const accuracy = correct / total;
  
  expect(accuracy).toBeGreaterThanOrEqual(0.5);
});

test('ImageClassifier: classifies all fake images as fake (with high probability)', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();
  const results = await classifier.classifyAllImages();

  const correct = results.fake.filter(r => r.correct).length;
  const total = results.fake.length;
  const accuracy = correct / total;
  
  expect(accuracy).toBeGreaterThanOrEqual(0.5);
});

test('ImageClassifier: overall accuracy should be high (80% expected)', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();
  const results = await classifier.classifyAllImages();

  expect(results.accuracy).toBeGreaterThanOrEqual(50);
});
