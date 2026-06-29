package com.mlp;

import java.util.Random;

public class MLP extends Model {

        // Parameters
        private double[][] W1, W2, W3;
        private double[] b1, b2, b3;

        // Gradients
        private double[][] dW1, dW2, dW3;
        private double[] db1, db2, db3;

        // Forward-pass cache (for backprop)
        private double[][] X;          // input batch
        private double[][] Z1, H1;     // hidden layer 1 pre-activation & activation
        private double[][] Z2, H2;     // hidden layer 2 pre-activation & activation
        private double[][] Z3, y_hat;     // output pre-activation & softmax ou
    public MLP() {

        // Allocate parameters
        W1 = new double[784][256];
        b1 = new double[256];
        W2 = new double[256][64];
        b2 = new double[64];
        W3 = new double[64][10];
        b3 = new double[10];

        // Allocate gradients (same shapes)
        dW1 = new double[784][256];
        db1 = new double[256];
        dW2 = new double[256][64];
        db2 = new double[64];
        dW3 = new double[64][10];
        db3 = new double[10];

        heInit(W1);
        heInit(W2);
        heInit(W3);

    }

    private static void heInit(double[][] W){
        Random rng = new Random();
        int nIn = W.length;
        double scale = Math.sqrt(2.0/ nIn);
        for (int i = 0; i < nIn; i++){
            for (int j = 0; j < W[0].length; j++){
                W[i][j] = rng.nextGaussian() * scale;
            }
        }
    }

    @Override
    public double[][] forward(double[][] batchX){
        X = batchX;
        Z1 = MatrixUtils.addBias(MatrixUtils.matMul(X, W1), b1);
        H1 = MatrixUtils.relu(Z1);
        Z2 = MatrixUtils.addBias(MatrixUtils.matMul(H1, W2), b2);
        H2 = MatrixUtils.relu(Z2);
        Z3 = MatrixUtils.addBias(MatrixUtils.matMul(H2, W3), b3);
        y_hat = MatrixUtils.softmax(Z3);
        return y_hat;
    }

    @Override
    public double loss(double[][] y_hat, double[][] y) {
        int N = y.length;
        int classes = y_hat[0].length;
        double loss = 0;
        for (int i = 0; i < N; i++){
            for (int j = 0; j < classes; j++){
                loss += -y[i][j] * Math.log(y_hat[i][j] + 1e-12);
            }
        }
        loss = loss / N;
        return loss;
    }
}
