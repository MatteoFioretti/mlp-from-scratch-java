package com.mlp;


import java.io.IOException;
public class Main {
    public static void main(String[] args) throws  IOException {
        MNISTDataModule trainData = new MNISTDataModule(128,
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte");

        int batchCount = 0;
        for (Batch batch: trainData){
            batchCount ++;
        }
        System.out.println("Total batches: " + batchCount);
    }
}

