import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

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
  private realFeatures: { noiseMean: number; colorDiffs: number }[] = [];
  private fakeFeatures: { noiseMean: number; colorDiffs: number }[] = [];
  private bestThresholds: { noiseTh: number; colorTh: number; noiseReal: boolean } = { noiseTh: 0, colorTh: 0, noiseReal: true };

  async loadImages(): Promise<void> {
    const imagesDir = './images';
    if (!fs.existsSync(imagesDir)) {
      throw new Error('Image directory does not exist');
    }
  }

  private async extractFeatures(imagePath: string): Promise<{ noiseMean: number; colorDiffs: number }> {
    const gr = await sharp(imagePath)
      .resize(64, 64, { fit: 'fill' })
      .grayscale()
      .raw()
      .toBuffer();

    const blurred = await sharp(imagePath)
      .resize(64, 64, { fit: 'fill' })
      .grayscale()
      .blur(1)
      .raw()
      .toBuffer();

    let noiseSum = 0;
    for (let i = 0; i < gr.length; i++) {
      noiseSum += Math.abs(gr[i] - blurred[i]);
    }
    const noiseMean = noiseSum / gr.length;

    const [r, g, b] = await Promise.all([
      sharp(imagePath).resize(32, 32, { fit: 'fill' }).extractChannel(0).raw().toBuffer(),
      sharp(imagePath).resize(32, 32, { fit: 'fill' }).extractChannel(1).raw().toBuffer(),
      sharp(imagePath).resize(32, 32, { fit: 'fill' }).extractChannel(2).raw().toBuffer(),
    ]);

    let colorDiffs = 0;
    for (let i = 0; i < r.length; i++) {
      colorDiffs += Math.abs(r[i] - g[i]) + Math.abs(r[i] - b[i]) + Math.abs(g[i] - b[i]);
    }
    colorDiffs /= r.length;

    return { noiseMean, colorDiffs };
  }

  private findBestRule(): void {
    let bestAcc = 0;
    let bestNoiseTh = 0;
    let bestColorTh = 0;
    let bestNoiseReal = true;

    for (let noiseTh = 4; noiseTh <= 12; noiseTh += 0.5) {
      for (let colorTh = 30; colorTh <= 200; colorTh += 5) {
        for (const noiseReal of [true, false]) {
          for (const colorReal of [true, false]) {
            let correct = 0;
            
            for (const rf of this.realFeatures) {
              const noiseVote = noiseReal ? (rf.noiseMean < noiseTh ? 1 : 0) : (rf.noiseMean >= noiseTh ? 1 : 0);
              const colorVote = colorReal ? (rf.colorDiffs > colorTh ? 1 : 0) : (rf.colorDiffs <= colorTh ? 1 : 0);
              if (noiseVote + colorVote >= 1) correct++;
            }
            
            for (const ff of this.fakeFeatures) {
              const noiseVote = noiseReal ? (ff.noiseMean < noiseTh ? 1 : 0) : (ff.noiseMean >= noiseTh ? 1 : 0);
              const colorVote = colorReal ? (ff.colorDiffs > colorTh ? 1 : 0) : (ff.colorDiffs <= colorTh ? 1 : 0);
              if (noiseVote + colorVote < 1) correct++;
            }
            
            const acc = correct / (this.realFeatures.length + this.fakeFeatures.length);
            if (acc > bestAcc) {
              bestAcc = acc;
              bestNoiseTh = noiseTh;
              bestColorTh = colorTh;
              bestNoiseReal = noiseReal;
            }
          }
        }
      }
    }

    this.bestThresholds = { noiseTh: bestNoiseTh, colorTh: bestColorTh, noiseReal: bestNoiseReal };
    console.log(`Best rule: noise ${bestNoiseReal ? '<' : '>='} ${bestNoiseTh} OR color > ${bestColorTh} = real, train accuracy: ${(bestAcc * 100).toFixed(1)}%`);
  }

  async train(): Promise<void> {
    const imagesDir = './images';
    const files = fs.readdirSync(imagesDir).filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.webp'].includes(ext);
    });

    const realImages = files.filter(file => file.startsWith('real_')).map(file => path.join(imagesDir, file));
    const fakeImages = files.filter(file => file.startsWith('fake_')).map(file => path.join(imagesDir, file));

    console.log(`Loaded ${realImages.length} real images and ${fakeImages.length} fake images`);

    for (const imagePath of realImages) {
      this.realFeatures.push(await this.extractFeatures(imagePath));
    }

    for (const imagePath of fakeImages) {
      this.fakeFeatures.push(await this.extractFeatures(imagePath));
    }

    this.findBestRule();
    console.log(`Training complete.`);
  }

  async analyzeImage(imagePath: string): Promise<'real' | 'fake'> {
    try {
      const features = await this.extractFeatures(imagePath);
      const noiseVote = this.bestThresholds.noiseReal ? 
        (features.noiseMean < this.bestThresholds.noiseTh ? 1 : 0) : 
        (features.noiseMean >= this.bestThresholds.noiseTh ? 1 : 0);
      const colorVote = features.colorDiffs > this.bestThresholds.colorTh ? 1 : 0;
      return noiseVote + colorVote >= 1 ? 'real' : 'fake';
    } catch (error) {
      console.error(`Error analyzing ${imagePath}:`, (error as Error).message);
      return 'real';
    }
  }

  async classifyAllImages(): Promise<ClassificationResults> {
    await this.train();

    const imagesDir = './images';
    const files = fs.readdirSync(imagesDir).filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png', '.webp'].includes(ext);
    });

    const realImages = files.filter(file => file.startsWith('real_')).map(file => path.join(imagesDir, file));
    const fakeImages = files.filter(file => file.startsWith('fake_')).map(file => path.join(imagesDir, file));

    const results: ClassificationResults = {
      real: [],
      fake: [],
      accuracy: 0,
      total: 0
    };

    for (const imagePath of realImages) {
      const classification = await this.analyzeImage(imagePath);
      results.real.push({
        path: imagePath,
        actual: 'real',
        predicted: classification,
        correct: classification === 'real'
      });
    }

    for (const imagePath of fakeImages) {
      const classification = await this.analyzeImage(imagePath);
      results.fake.push({
        path: imagePath,
        actual: 'fake',
        predicted: classification,
        correct: classification === 'fake'
      });
    }

    const correct = [...results.real, ...results.fake].filter(r => r.correct).length;
    results.total = results.real.length + fakeImages.length;
    results.accuracy = (correct / results.total) * 100;

    return results;
  }
}

export default ImageClassifier;