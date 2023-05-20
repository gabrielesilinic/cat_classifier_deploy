class CatClassifier {
    constructor(modelPath) {
        this.modelPath = modelPath;
    }

    async loadModel() {
        this.model = await tf.loadLayersModel(this.modelPath);
    }

    async predict(imgElement) {
        if (!this.model) {
            console.error('Model not loaded, please call `loadModel` first.');
            return;
        }

        // Preprocess the image: resize to 200x200 pixels, grayscale, expand dimensions, normalize
        let tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([200, 200])
            .mean(2)
            .expandDims(2)
            .expandDims()
            .div(255.0);

        // Run the model on the tensor
        let prediction = this.model.predict(tensor);

        // The model outputs a score between 0 and 1 (probability of being a cat)
        let score = prediction.dataSync()[0];

        return score > 0.5 ? 'Not a cat' : 'Cat';
    }
}
