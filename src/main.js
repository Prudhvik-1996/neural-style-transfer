import * as tf from '@tensorflow/tfjs';

// Configuration
const CONFIG = {
  imageSize: 256,
  contentLayer: 'block5_conv2',
  styleLayers: ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
  contentWeight: 1.0,
  styleWeight: 0.01,
  tvWeight: 0.0001,
  iterations: 1000,
  learningRate: 0.02
};

let model = null;

// DOM Elements
const contentInput = document.getElementById('contentImage');
const styleInput = document.getElementById('styleImage');
const startButton = document.getElementById('startButton');
const progressDiv = document.getElementById('progress');
const contentCanvas = document.getElementById('contentCanvas');
const styleCanvas = document.getElementById('styleCanvas');
const outputCanvas = document.getElementById('outputCanvas');

// Load VGG19 model
async function loadModel() {
  try {
    console.log("Loading model...");
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/paulsp94/tfjs_vgg19_imagenet/refs/heads/master/model/model.json');
    console.log("Model loaded successfully.");
    startButton.disabled = false;
  } catch (error) {
    console.error("Failed to load model:", error);
  }
}

// Load and process image
async function loadImage(file) {
  return new Promise((resolve) => {
      const img = new Image();
      img.onload = async () => {
          await img.decode();
          resolve(img);
      };
      img.src = URL.createObjectURL(file);
  });
}

// Resize and preprocess image for VGG19
function preprocessImage(img, targetSize) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  canvas.width = targetSize;
  canvas.height = targetSize;
  ctx.drawImage(img, 0, 0, targetSize, targetSize);

  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  let tensor = tf.browser.fromPixels(imageData)
      .toFloat()
      .expandDims(); // Shape: [1, targetSize, targetSize, 3]

  console.log("Processed Image Tensor Shape: " + JSON.stringify(tensor.shape));

  // Normalize for VGG19
  tensor = tf.sub(
      tf.mul(tensor, 255), 
      tf.tensor1d([103.939, 116.779, 123.68]).reshape([1, 1, 1, 3])
  );

  return tensor;
}

// Compute Gram matrix
function gramMatrix(activations) {
  return tf.tidy(() => {
    const channels = activations.shape[3];
    const a = tf.reshape(activations, [-1, channels]);
    return tf.div(tf.matMul(a, a, true), tf.scalar(a.shape[0]));
  });
}

// Loss functions
function contentLoss(content, generated) {
  return tf.mean(tf.square(tf.sub(content, generated)));
}

function styleLoss(style, generated) {
  return tf.mean(tf.square(tf.sub(gramMatrix(style), gramMatrix(generated))));
}

function totalVariationLoss(image) {
  return tf.tidy(() => {
    const horizontalDiff = tf.square(image.slice([0, 1, 0, 0], [-1, -1, -1, -1]).sub(image.slice([0, 0, 0, 0], [-1, -1, -1, -1])));
    const verticalDiff = tf.square(image.slice([1, 0, 0, 0], [-1, -1, -1, -1]).sub(image.slice([0, 0, 0, 0], [-1, -1, -1, -1])));
    return tf.mean(tf.add(horizontalDiff, verticalDiff));
  });
}

// Style transfer execution
async function styleTransfer(contentImage, styleImage) {
  tf.engine().startScope();
  try {
    // Determine the smaller dimension and resize both images accordingly
    const minSize = Math.min(contentImage.width, contentImage.height, styleImage.width, styleImage.height);
    console.log(`Using size: ${minSize}x${minSize}`);

    const contentTensor = preprocessImage(contentImage, minSize);
    const styleTensor = preprocessImage(styleImage, minSize);
    const generatedImage = tf.variable(contentTensor.clone());
    
    const contentFeatures = model.predict(contentTensor);
    const styleFeatures = model.predict(styleTensor);
    
    const optimizer = tf.train.adam(CONFIG.learningRate);
    
    for (let i = 0; i < CONFIG.iterations; i++) {
      optimizer.minimize(() => {
        const generatedFeatures = model.predict(generatedImage);
        
        const cLoss = contentLoss(contentFeatures, generatedFeatures).mul(CONFIG.contentWeight);
        const sLoss = styleLoss(styleFeatures, generatedFeatures).mul(CONFIG.styleWeight);
        const tvLoss = totalVariationLoss(generatedImage).mul(CONFIG.tvWeight);
        
        const totalLoss = tf.add(tf.add(cLoss, sLoss), tvLoss);
        
        if (i % 10 === 0) {
          progressDiv.textContent = `Iteration ${i}/${CONFIG.iterations}, Loss: ${totalLoss.dataSync()[0].toFixed(2)}`;
        }
        
        return totalLoss;
      });
      
      if (i % 50 === 0) {
        await tf.browser.toPixels(tf.sigmoid(generatedImage.squeeze()), outputCanvas);
      }
    }
    
    await tf.browser.toPixels(tf.sigmoid(generatedImage.squeeze()), outputCanvas);
  } catch (error) {
    console.error('Style transfer failed:', error);
    progressDiv.textContent = 'Style transfer failed: ' + error.message;
  } finally {
    tf.engine().endScope();
  }
}

// Event listeners
contentInput.addEventListener('change', async (e) => {
  const img = await loadImage(e.target.files[0]);
  contentCanvas.getContext('2d').drawImage(img, 0, 0, contentCanvas.width, contentCanvas.height);
});

styleInput.addEventListener('change', async (e) => {
  const img = await loadImage(e.target.files[0]);
  styleCanvas.getContext('2d').drawImage(img, 0, 0, styleCanvas.width, styleCanvas.height);
});

startButton.addEventListener('click', async () => {
  if (!contentInput.files[0] || !styleInput.files[0]) {
    alert('Please select both content and style images');
    return;
  }
  
  startButton.disabled = true;
  try {
    const contentImage = await loadImage(contentInput.files[0]);
    const styleImage = await loadImage(styleInput.files[0]);
    await styleTransfer(contentImage, styleImage);
  } catch (error) {
    console.error('Error:', error);
    progressDiv.textContent = 'Error: ' + error.message;
  }
  startButton.disabled = false;
});

// Load model on page load
loadModel().catch(console.error);
