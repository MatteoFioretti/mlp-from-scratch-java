package com.mlp;

public class Batch {
    private final double[][] X;
    private final double [][]y;

    public Batch(double[][] X, double[][] y){
        this.X = X;
        this.y = y;
    }

    public double[][] getX(){
        return X;
    }

    public double[][] gety(){
        return y;
    }
}
