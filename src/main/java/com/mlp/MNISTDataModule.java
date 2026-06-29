package com.mlp;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

public class MNISTDataModule extends DataModule {
    private double[][] X;
    private double[][] y;

    public MNISTDataModule(int batchSize, String imagePath, String labelPath) throws IOException {
        super(batchSize);
        this.X = loadImages(imagePath);
        this.y = loadLabels(labelPath);
    }
    public double[][] getX() { return X; }
    public double[][] getY() { return y; }
    @Override
    public Iterator<Batch> iterator(){
        return new BatchIterator();
    }

    private class BatchIterator implements Iterator<Batch>{
        private int currentIndex = 0;

        public boolean hasNext(){
            return currentIndex < X.length;
        }

        public Batch next(){
            int currBatchSize = Math.min(batchSize, X.length - currentIndex);
            double[][] X_batch = Arrays.copyOfRange(X, currentIndex, currentIndex + currBatchSize);
            double[][] y_batch = Arrays.copyOfRange(y,currentIndex, currentIndex+currBatchSize);
            currentIndex += currBatchSize;
            return new Batch(X_batch, y_batch);
        }


    }
    private double[][] loadImages(String path) throws IOException{
        // opening the file
        try(DataInputStream imagePipe = new DataInputStream(
                new BufferedInputStream(
                        new FileInputStream(path)))) {

                    int pixels = 784;
                    int magicNumber = imagePipe.readInt();
                    int count = imagePipe.readInt();
                    int rows = imagePipe.readInt();
                    int cols = imagePipe.readInt();

                    if (magicNumber != 2051) {
                        throw new IOException("Invalid image file magic number: " + magicNumber);
                    }

                    double[][] result = new double[count][pixels];

                    for (int i = 0; i < count; i++){
                        for (int j = 0; j < pixels; j++) {
                            result[i][j] = imagePipe.readUnsignedByte() / 255.0;
                        }
                    }
                    return result;
        }
    }

    private double[][] loadLabels (String path) throws IOException{
        try(DataInputStream labelPipe = new DataInputStream(
                new BufferedInputStream(
                        new FileInputStream(path)))) {
           int classes = 10;
           int magicNumber = labelPipe.readInt();
           int count = labelPipe.readInt();

            if (magicNumber != 2049) {
                throw new IOException("Invalid label file magic number: " + magicNumber);
            }

            double[][] result = new double[count][classes];

            for (int i = 0; i < count; i++){
               int label = labelPipe.readUnsignedByte();
               result[i][label] = 1;
           }

            return result;
        }
    }
}
