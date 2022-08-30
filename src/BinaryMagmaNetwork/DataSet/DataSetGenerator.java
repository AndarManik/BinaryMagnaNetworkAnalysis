package BinaryMagmaNetwork;

import NeuralNetwork.BiasManager;
import NeuralNetwork.HyperParameters;

import java.util.ArrayList;

public class DataSetGenerator {


    public static void main(String[] args) {

        while (true) {
            wait(1000);
            if (Thread.activeCount() < 18) {
                new Thread(DataSetGenerator::createAndSave).start();
            }
        }
    }


    private static void createAndSave() {
        BiasManager works = works(HyperParameters.FAST_HYPER);
        System.out.println(works);
        assert works != null;
        DataSetWriter.writeTo(works);
    }

    static final ArrayList<double[]> INPUT_SPACE = binaryProductSpace(2);
    static final ArrayList<double[]> OUTPUT_SPACE = binaryProductSpace(4);

    private static BiasManager works(HyperParameters hyper) {
        for (int i = 0; i < hyper.tries(); i++) {
            BiasManager bm = train(hyper.epocMag(), hyper.rate());
            if (proof(bm) < hyper.cutOff())
                return bm;
        }
        return null;
    }

    /**
     * Sums of the errors of the network at its current state
     *
     * @return Sum of errors
     */
    private static double proof(BiasManager bm) {
        double sum = 0;
        for (double d : getScoreList(bm))
            sum += d;
        System.out.println(sum);
        return sum;
    }

    /**
     * Computes the error of the Bias Manager for every tasks
     *
     * @return Array storing the errors for every task
     */
    private static double[] getScoreList(BiasManager bm) {
        double[] scoreList = new double[16];
        for (int task = 0; task < 16; task++)
            scoreList[task] = getTaskError(bm, task);
        return scoreList;
    }

    /**
     * Computes the error for a single task
     *
     * @param task Current task for the Bias Manager
     * @return Error for the current task
     */
    private static double getTaskError(BiasManager bm, int task) {
        bm.setBias(task);
        double[] currentOp = OUTPUT_SPACE.get(task);
        double error = 0;
        for (int i = 0; i < currentOp.length; i++)
            error += Math.abs((bm.calc(INPUT_SPACE.get(i))[0] - currentOp[i]));
        return error;
    }


    /**
     * initialize a new bnn and train it epocMag times
     * return the final error of the bnn
     *
     * @param epocMag 10^epocMag number of epoc iterations
     * @param rate    learning rate for the bnn
     * @return final error of the bnn
     */
    private static BiasManager train(double epocMag, double rate) {
        BiasManager bm = new BiasManager(new int[]{2, 3, 1}, 16);
        setFinalBiasZero(bm);
        for (int epoc = 0; epoc < Math.pow(10, epocMag); epoc++)
            singlePass(bm, rate, epoc % 16);
        return bm;
    }

    /**
     * a single pass through an entire task space
     *
     * @param rate learning rate for the bnn
     * @param task which task and biases to use on the bnn
     */
    private static void singlePass(BiasManager bm, double rate, int task) {
        bm.setBias(task);
        double[] currentOp = OUTPUT_SPACE.get(task);
        for (int i = 0; i < INPUT_SPACE.size(); i++)
            bm.back(INPUT_SPACE.get(i), new double[]{currentOp[i]});
        bm.update(rate);
        setFinalBiasZero(bm);
    }


    /**
     * Computes an ArrayList which stores every binary number of a certain digit count
     *
     * @param dimension Number of digits
     * @return ArrayList of Double[] which contain the individual binary digits
     */
    public static ArrayList<double[]> binaryProductSpace(double dimension) {
        ArrayList<double[]> productSpace = new ArrayList<>();
        for (int i = 0; i < Math.pow(2, dimension); i++)
            productSpace.add(getBinaryRepresentation(dimension, i));
        return productSpace;
    }

    /**
     * Convert number to binary and stores the digits in an array
     *
     * @param dimension Number of digits
     * @param number    Current number to be converted
     * @return Double[] which stores the individual digits
     */
    private static double[] getBinaryRepresentation(double dimension, int number) {
        double[] output = new double[(int) dimension];
        int remainder = number;
        for (int j = 0; j < dimension; j++, remainder /= 2)
            output[j] = (remainder % 2 - 0.5) * 2;
        return output;
    }

    private static void setFinalBiasZero(BiasManager bm) {
        bm.network.get(bm.network.size() - 1).bias[0] = 0;
    }

    public static void wait(int ms)
    {
        try
        {
            Thread.sleep(ms);
        }
        catch(InterruptedException ex)
        {
            Thread.currentThread().interrupt();
        }
    }
}
