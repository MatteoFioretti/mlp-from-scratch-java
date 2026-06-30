package com.mlp;

import java.util.Random;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

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

    @Override
    public void backward(double[][] y){
        // First block
        double[][] dZ3 = MatrixUtils.subtract(y_hat , y);
        dW3 = MatrixUtils.matMul(MatrixUtils.transpose(H2), dZ3);
        db3 = MatrixUtils.sumColumns(dZ3);
        double[][] dH2 = MatrixUtils.matMul(dZ3, MatrixUtils.transpose(W3));
        // Second block
        double[][] dZ2 = MatrixUtils.multiply(dH2, MatrixUtils.reluDerivative(Z2));
        dW2 = MatrixUtils.matMul(MatrixUtils.transpose(H1), dZ2);
        db2 = MatrixUtils.sumColumns(dZ2);
        double[][] dH1 = MatrixUtils.matMul(dZ2, MatrixUtils.transpose(W2));
        // Third block
        double[][] dZ1 = MatrixUtils.multiply(dH1, MatrixUtils.reluDerivative(Z1));
        dW1 = MatrixUtils.matMul(MatrixUtils.transpose(X), dZ1);
        db1 = MatrixUtils.sumColumns(dZ1);
    }



    @Override
    public void step(double learningRate){
        MatrixUtils.updateInPlace(W1, dW1, learningRate );
        MatrixUtils.updateInPlace(b1, db1, learningRate);
        MatrixUtils.updateInPlace(W2, dW2, learningRate );
        MatrixUtils.updateInPlace(b2, db2, learningRate);
        MatrixUtils.updateInPlace(W3, dW3, learningRate );
        MatrixUtils.updateInPlace(b3, db3, learningRate);
    }

    // ===== Persistence =====

    public void save(String path) throws IOException {
        // PrintWriter wrapped over a BufferedWriter/FileWriter: lets us write text efficiently.
        // try-with-resources auto-closes the file when done (same idiom as the MNIST loader).
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {
            writeMatrix(writer, W1);
            writeVector(writer, b1);
            writeMatrix(writer, W2);
            writeVector(writer, b2);
            writeMatrix(writer, W3);
            writeVector(writer, b3);
        }
    }

    public void load(String path) throws IOException {
        // Scanner reads tokens from the file; nextDouble() pulls and parses one number at a time,
        // automatically skipping the whitespace between numbers.
        try (Scanner scanner = new Scanner(new File(path))) {
            scanner.useLocale(java.util.Locale.US);   // force dot-decimal parsing
            readMatrix(scanner, W1);
            readVector(scanner, b1);
            readMatrix(scanner, W2);
            readVector(scanner, b2);
            readMatrix(scanner, W3);
            readVector(scanner, b3);
        }
    }

// --- write helpers ---

    private static void writeMatrix(PrintWriter writer, double[][] M) {
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                writer.print(new java.math.BigDecimal(M[i][j]).toPlainString());
                writer.print(" ");
            }
            writer.println();
        }
    }

    private static void writeVector(PrintWriter writer, double[] v) {
        for (int i = 0; i < v.length; i++) {
            writer.print(new java.math.BigDecimal(v[i]).toPlainString());
            writer.print(" ");
        }
        writer.println();
    }

// --- read helpers ---

    private static void readMatrix(Scanner scanner, double[][] M) {
        // We know M's exact dimensions, so we read exactly that many numbers, in order.
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                M[i][j] = scanner.nextDouble();
            }
        }
    }

    private static void readVector(Scanner scanner, double[] v) {
        for (int i = 0; i < v.length; i++) {
            v[i] = scanner.nextDouble();
        }
    }
}


