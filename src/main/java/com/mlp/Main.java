package com.mlp;


import java.io.IOException;
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
        MNISTDataModule trainData = new MNISTDataModule(128,
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte");

        MNISTDataModule testData = new MNISTDataModule(128,
                "data/t10k-images.idx3-ubyte",
                "data/t10k-labels.idx1-ubyte");

        MLP model = new MLP();

        Trainer trainer = new Trainer(model, trainData, 0.001, 10);

        trainer.train();
        trainer.evaluate(testData);
    }
}

