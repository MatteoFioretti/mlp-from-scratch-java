package com.mlp;


public abstract class DataModule implements Iterable<Batch> {
    protected int batchSize;
    public DataModule(int batchSize){
        this.batchSize = batchSize;
    }

}
