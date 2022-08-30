package BinaryMagmaNetwork;

import NeuralNetwork.BiasManager;
import NeuralNetwork.Layer;

import java.io.*;
import java.util.ArrayList;

public class DataSetWriter {
    private static DataSetWriter dataSetWriter = null;
    static final File file = new File("src/BinaryMagmaNetwork/DataSet/BMNDataSet.txt");
    static final FileWriter fileWriter;

    static {
        try {
            fileWriter = new FileWriter(file, true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }));
    }

    public synchronized static void writeTo(BiasManager biasManager) {
        try {
            fileWriter.append("##\n");
            for (Layer layer : biasManager.network)
                for (double[] doubles : layer.weight)
                    for (double aDouble : doubles)
                        fileWriter.append(String.valueOf(aDouble)).append("\n");

            for (ArrayList<double[]> bias : biasManager.biases)
                for (double[] bia : bias)
                    for (double v : bia)
                        fileWriter.append(String.valueOf(v)).append("\n");


        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
