package com.mlp;

public class MatrixUtils {
    public static double[][] matMul(double[][] X, double[][] W) {
        int n = X.length;
        int m = X[0].length;
        int p = W[0].length;
        double[][] M = new double[n][p];
        for (int i = 0; i < n; i++){
            for (int j = 0; j < p; j++){
                for (int k = 0; k < m; k++ ){
                    M[i][j] += X[i][k] * W[k][j];
                }
            }
        }
        return M;
    }

    public static double[][] addBias(double[][] M, double[] b) {
        double[][] Z = new double[M.length][M[0].length];
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                Z[i][j] = M[i][j] + b[j];
            }
        }
        return Z;
    }
    public static double[][] relu(double[][] Z) {
        double[][] H = new double[Z.length][Z[0].length];
        for (int i = 0; i < H.length; i++){
            for (int j = 0; j < H[0].length; j++){
                H[i][j] = Math.max(0, Z[i][j]);
            }
        }
        return H;
    }

    public static double[][] softmax(double[][] W) {
        double[][] O = new double[W.length][W[0].length];
        for (int i = 0; i < W.length; i++) {
            double[] row = W[i];
            double max = row[0];
            for (int j = 0; j < row.length; j++) {
                max = Math.max(max, row[j]);
            }
            double sum = 0;
            for (int j = 0; j < row.length; j++) {
                O[i][j] = Math.exp(row[j] - max);  // write to O, read from row
                sum += O[i][j];
            }
            for (int j = 0; j < row.length; j++) {
                O[i][j] = O[i][j] / sum;
            }
        }
        return O;
    }

    public static double[][] transpose(double[][] W){
        int rows = W.length;
        int cols = W[0].length;
        double[][] T = new double[cols][rows];
        for (int i = 0; i < cols; i++){
            for (int j = 0; j < rows; j++){
                T[i][j] = W[j][i];
            }
        }
        return T;
    }

    public static double[][] subtract(double[][] A, double[][] B) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    public static double[][] multiply(double[][] A, double[][] B) {
        double[][] C = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = A[i][j] * B[i][j];
            }
        }
        return C;
    }

    public static double[] sumColumns(double[][] dZ) {
        int rows = dZ.length;
        int cols = dZ[0].length;
        double[] result = new double[cols];
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                result[j] += dZ[i][j];
            }
        }
        return result;
    }

    public static double[][] reluDerivative(double[][] Z) {
        double[][] D = new double[Z.length][Z[0].length];
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[0].length; j++) {
                D[i][j] = Z[i][j] > 0 ? 1.0 : 0.0;
            }
        }
        return D;
    }

    public static void updateInPlace(double[][] W, double[][] dW, double lr) {
        for (int i = 0; i < W.length; i++){
            for (int j = 0; j < W[0].length; j++){
                W[i][j] -= lr * dW[i][j];
            }
        }
    }
    public static void updateInPlace(double[] b, double[] db, double lr) {
        for (int i = 0; i < b.length; i++){
            b[i] -= lr * db[i];
        }
    }

    public static int argmax(double[] row){
        int idx = 0;
        for (int i = 1; i < row.length; i++){
            if (row[i] > row[idx]){
                idx = i;
            }
        }
        return idx;
    }

}
