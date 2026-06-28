package com.mlp;
// Field: batchsize
// it implements Iterable<Batch> which forces it to provide and iterator() method

public abstract class DataModule implements Iterable<Batch> {
    protected int batchSize;
    public DataModule(int batchSize){
        this.batchSize = batchSize;
    }

}
