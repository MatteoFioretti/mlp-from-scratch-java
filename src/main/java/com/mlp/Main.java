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
    public static void main(String[] args) throws  IOException, InterruptedException {
        MNISTDataModule trainData = new MNISTDataModule(128,
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte");

        // MLP test
        Batch first = trainData.iterator().next();
        double[][] batchX = first.getX();
        double[][] label = first.gety();
        MLP model = new MLP();
        double[][] preds = model.forward(batchX);
        double[] row = preds[0];
        double sum = 0;

        /*
        for (int i = 0; i < row.length; i++){
            sum += row[i];
        }
        System.out.println(sum);
        */
        double loss = model.loss(preds, label);
        System.out.println("Initial loss value: " + loss);

        double learningRate = 0.001;
        for (int i = 0; i < 50; i++) {
            double[][] y_hat = model.forward(batchX);
            double l = model.loss(y_hat, label);
            System.out.println("Step " + i + " loss: " + l);
            model.backward(label);
            model.step(learningRate);
        }

    }
}

