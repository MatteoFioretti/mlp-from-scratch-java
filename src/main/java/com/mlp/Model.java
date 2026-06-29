package com.mlp;

public abstract class Model {

    public abstract double[][] forward(double[][] batchX);

    public abstract double loss(double[][] y_hat, double[][] y);

    public  abstract void backward(double[][] y);

    public abstract void step(double learningRate);
}
