package com.mlp;


import java.io.IOException;
import java.io.File;
public class Main {

    public static void printImage(double[] flatImage, double[] label) throws InterruptedException {
        int imgSize = 28;
        for (int i = 0; i < imgSize; i++){
            for (int j = 0; j < imgSize; j++){
                double pixel = flatImage[(i * 28) + j];
                if (pixel < 0.25) {
                    System.out.print("  ");
                } else if (pixel < 0.50) {
                    System.out.print(". ");
                } else if (pixel < 0.75) {
                    System.out.print("+ ");
                } else {
                    System.out.print("# ");
                }
            }
            System.out.println();
            Thread.sleep(150); // pauses for 50 milliseconds
        }
        int digit = 0;
        for (int k = 0; k < label.length; k++) {
            if (label[k] == 1.0) {
                digit = k;
            }
        }
        System.out.println("label: " + digit);
    }
    public static void main(String[] args) throws IOException, InterruptedException {
        String weightsPath = "weights.txt";

        MNISTDataModule trainData = new MNISTDataModule(128,
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte");
        MNISTDataModule testData = new MNISTDataModule(128,
                "data/t10k-images.idx3-ubyte",
                "data/t10k-labels.idx1-ubyte");

        MLP model = new MLP();

        // Train only if no saved weights exist yet; otherwise load them.
        if (new File(weightsPath).exists()) {
            System.out.println("Loading saved weights...");
            model.load(weightsPath);
        } else {
            System.out.println("No saved weights found — training...");
            Trainer trainer = new Trainer(model, trainData, 0.001, 10);
            trainer.train();
            model.save(weightsPath);
            System.out.println("Weights saved to " + weightsPath);
        }

        // Evaluate the (loaded or freshly trained) model on the test set.
        Trainer evaluator = new Trainer(model, trainData, 0.001, 10);
        double testAcc = evaluator.evaluate(testData);

        // Demo: classify a few test digits visually.
        System.out.println("\n===== DEMO: classifying test digits =====");
        double[][] testX = testData.getX();
        double[][] testY = testData.getY();

        int numToShow = 5;
        for (int i = 0; i < numToShow; i++) {
            printImage(testX[i], testY[i]);
            double[][] single = { testX[i] };          // wrap one image as a [1][784] batch
            double[][] pred = model.forward(single);
            int predicted = MatrixUtils.argmax(pred[0]);
            int actual = MatrixUtils.argmax(testY[i]);
            System.out.println("Predicted: " + predicted + "  |  Actual: " + actual);
            System.out.println("------------------------------------------");
            Thread.sleep(800);
        }
    }
}

