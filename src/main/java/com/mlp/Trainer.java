package com.mlp;

public class Trainer {

    private Model model;
    private DataModule data;
    private double learningRate;
    private int epochs;

    public Trainer(Model model, DataModule data, double learningRate, int epochs ){
        this.model = model;
        this.data = data;
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    private double accuracy(double[][] y_hat, double[][] y){
        double correct = 0;
        for (int i = 0; i < y.length; i++){
            if (MatrixUtils.argmax(y_hat[i]) == MatrixUtils.argmax(y[i])){
                correct += 1;
            }
        }
        return correct/y.length;
    }

    public void train(){
        for (int epoch = 0; epoch < this.epochs; epoch++){
            double loss = 0;
            double accuracy = 0;
            int batchCounter = 0;
            for (Batch batch: data){
                batchCounter += 1;

                double[][] batchX = batch.getX();
                double[][] y = batch.gety();

                double[][] y_hat = model.forward(batchX);
                double batchLoss = model.loss(y_hat, y);
                double batchAccuracy = accuracy(y_hat, y);

                loss += batchLoss;
                accuracy += batchAccuracy;

                model.backward(y);
                model.step(learningRate);
            }
            double avgLoss = loss/batchCounter;
            double avgAccuracy = accuracy/batchCounter;

            System.out.printf("Epoch %d — loss: %.4f, accuracy: %.2f%%%n",
                    epoch + 1, avgLoss, avgAccuracy * 100);
        }
    }

    public double evaluate(DataModule testData){
        double loss = 0;
        double accuracy = 0;
        int batchCounter = 0;
        for (Batch batch: testData){
            batchCounter += 1;

            double[][] batchX = batch.getX();
            double[][] y = batch.gety();
            double[][] y_hat = model.forward(batchX);
            double batchLoss = model.loss(y_hat, y);
            double batchAccuracy = accuracy(y_hat, y);

            loss += batchLoss;
            accuracy += batchAccuracy;
        }
        double avgLoss = loss/batchCounter;
        double avgAccuracy = accuracy/batchCounter;

        System.out.printf("Test — loss: %.4f, accuracy: %.2f%%%n",
                avgLoss, avgAccuracy * 100);
        return avgAccuracy;
    }
}
