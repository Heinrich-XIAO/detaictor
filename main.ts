import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import FFT from 'fft.js';

interface ClassificationResult {
  path: string;
  actual: 'real' | 'fake';
  predicted: 'real' | 'fake';
  correct: boolean;
}

interface ClassificationResults {
  real: ClassificationResult[];
  fake: ClassificationResult[];
  accuracy: number;
  total: number;
}

class ImageClassifier {
  realImages: string[] = [];
  fakeImages: string[] = [];
  realRatios: number[] = [];
  fakeRatios: number[] = [];
  avgReal: number = 0;
  avgFake: number = 0;

  async loadImages(): Promise<void> {
    const imagesDir = './images';

    if (!fs.existsSync(imagesDir)) {
      throw new Error('Image directory does not exist');
    }

    const files = fs.readdirSync(imagesDir).filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.webp'].includes(ext);
    });

    this.realImages = files.filter(file => file.startsWith('real_')).map(file => path.join(imagesDir, file));
    this.fakeImages = files.filter(file => file.startsWith('fake_')).map(file => path.join(imagesDir, file));

    console.log(`Loaded ${this.realImages.length} real images and ${this.fakeImages.length} fake images`);
  }

  async extractNoise(imagePath: string): Promise<Buffer> {
    const resized = await sharp(imagePath)
      .resize(256, 256, { fit: 'fill' })
      .grayscale()
      .raw()
      .toBuffer();

    const blurred = await sharp(imagePath)
      .resize(256, 256, { fit: 'fill' })
      .grayscale()
      .blur(3)
      .raw()
      .toBuffer();

    const noise = Buffer.alloc(resized.length);
    for (let i = 0; i < resized.length; i++) {
      noise[i] = Math.abs(resized[i] - blurred[i]);
    }

    return noise;
  }

  computeFFT(noise: Buffer): number[] {
    const size = 256;
    const fft = new FFT(size);
    
    const rowFFTs: number[][] = [];
    for (let row = 0; row < size; row++) {
      const rowData = Array.from(noise.slice(row * size, (row + 1) * size));
      const out = fft.createComplexArray();
      fft.realTransform(out, rowData);
      fft.completeSpectrum(out);
      rowFFTs.push(out);
    }
    
    const magnitudes = new Array<number>(size * size);
    
    for (let col = 0; col < size; col++) {
      const colData = new Array<number>(size);
      for (let row = 0; row < size; row++) {
        colData[row] = rowFFTs[row][col * 2];
      }
      
      const out = fft.createComplexArray();
      fft.realTransform(out, colData);
      fft.completeSpectrum(out);
      
      for (let row = 0; row < size; row++) {
        const real = out[row * 2];
        const imag = out[row * 2 + 1];
        magnitudes[row * size + col] = Math.sqrt(real * real + imag * imag);
      }
    }
    
    return magnitudes;
  }

  extractFeatures(magnitudes: number[]): number[] {
    const features: number[] = [];
    const size = 256;
    
    const centerX = size / 2;
    const centerY = size / 2;
    
    let lowFreqSum = 0, midFreqSum = 0, highFreqSum = 0;
    let lowCount = 0, midCount = 0, highCount = 0;
    
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = y * size + x;
        if (idx >= magnitudes.length) continue;
        
        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        
        if (dist < 30) {
          lowFreqSum += magnitudes[idx];
          lowCount++;
        } else if (dist < 80) {
          midFreqSum += magnitudes[idx];
          midCount++;
        } else if (dist < 120) {
          highFreqSum += magnitudes[idx];
          highCount++;
        }
      }
    }
    
    features.push(lowFreqSum / (lowCount || 1));
    features.push(midFreqSum / (midCount || 1));
    features.push(highFreqSum / (highCount || 1));
    
    let maxMag = 0, meanMag = 0;
    for (const mag of magnitudes) {
      if (mag > maxMag) maxMag = mag;
      meanMag += mag;
    }
    meanMag /= magnitudes.length;
    
    features.push(maxMag);
    features.push(meanMag);
    
    let variance = 0;
    for (const mag of magnitudes) {
      variance += (mag - meanMag) ** 2;
    }
    variance /= magnitudes.length;
    features.push(Math.sqrt(variance));
    
    return features;
  }

  computeHighFreqRatio(features: number[]): number {
    return features[2] / (features[0] + features[1] + features[2] + 1e-10);
  }

  async train(): Promise<void> {
    this.realRatios = [];
    this.fakeRatios = [];

    for (const imagePath of this.realImages) {
      const noise = await this.extractNoise(imagePath);
      const magnitudes = this.computeFFT(noise);
      const features = this.extractFeatures(magnitudes);
      this.realRatios.push(this.computeHighFreqRatio(features));
    }

    for (const imagePath of this.fakeImages) {
      const noise = await this.extractNoise(imagePath);
      const magnitudes = this.computeFFT(noise);
      const features = this.extractFeatures(magnitudes);
      this.fakeRatios.push(this.computeHighFreqRatio(features));
    }

    this.avgReal = this.realRatios.reduce((a, b) => a + b, 0) / this.realRatios.length;
    this.avgFake = this.fakeRatios.reduce((a, b) => a + b, 0) / this.fakeRatios.length;

    console.log(`Training complete. Real avg: ${this.avgReal.toFixed(4)}, Fake avg: ${this.avgFake.toFixed(4)}`);
  }

  async analyzeImage(imagePath: string): Promise<'real' | 'fake'> {
    try {
      const noise = await this.extractNoise(imagePath);
      const magnitudes = this.computeFFT(noise);
      const features = this.extractFeatures(magnitudes);
      
      const ratio = this.computeHighFreqRatio(features);
      
      const distToReal = Math.abs(ratio - this.avgReal);
      const distToFake = Math.abs(ratio - this.avgFake);
      
      return distToReal < distToFake ? 'real' : 'fake';
    } catch (error) {
      console.error(`Error analyzing ${imagePath}:`, (error as Error).message);
      return 'real';
    }
  }

  async classifyAllImages(): Promise<ClassificationResults> {
    await this.train();

    const results: ClassificationResults = {
      real: [],
      fake: [],
      accuracy: 0,
      total: 0
    };

    for (const imagePath of this.realImages) {
      const classification = await this.analyzeImage(imagePath);
      results.real.push({
        path: imagePath,
        actual: 'real',
        predicted: classification,
        correct: classification === 'real'
      });
    }

    for (const imagePath of this.fakeImages) {
      const classification = await this.analyzeImage(imagePath);
      results.fake.push({
        path: imagePath,
        actual: 'fake',
        predicted: classification,
        correct: classification === 'fake'
      });
    }

    const correct = [...results.real, ...results.fake].filter(r => r.correct).length;
    results.total = results.real.length + this.fakeImages.length;
    results.accuracy = (correct / results.total) * 100;

    return results;
  }
}

export default ImageClassifier;