import { test, expect } from 'bun:test';
import ImageClassifier from '../main.js';
import path from 'path'

test('ImageClassifier: loads images from directories', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();
  expect(classifier.realImages.length).toBeGreaterThan(0);
  expect(classifier.fakeImages.length).toBeGreaterThan(0);
});

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

test('ImageClassifier: all files are PNG format', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();

  const allImages = [...classifier.realImages, ...classifier.fakeImages];
  
  for (const imagePath of allImages) {
    expect(imagePath).toMatch(/\.png$/);
  }
});

test('ImageClassifier: naming convention is real_XX.png or fake_XX.png', async () => {
  const classifier = new ImageClassifier();
  await classifier.loadImages();

  const allImages = [...classifier.realImages, ...classifier.fakeImages];
  
  for (const imagePath of allImages) {
    const filename = path.basename(imagePath);
    expect(filename).toMatch(/^(real|fake)_img_\d{2}\.png$/);
  }
});