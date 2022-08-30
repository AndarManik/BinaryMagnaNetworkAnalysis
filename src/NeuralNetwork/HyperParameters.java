package BinaryMagmaNetwork;

public record HyperParameters(int tries, double cutOff, double epocMag, double rate){
    static final HyperParameters FAST_HYPER = new HyperParameters(100,  0.05, 6.5, 0.05);
}