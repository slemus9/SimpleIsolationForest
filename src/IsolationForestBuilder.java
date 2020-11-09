import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.function.Function;
import java.util.stream.DoubleStream;

/**
 * Implementation of Isolation Forest based on the document:
 * https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest
 */
public class IsolationForestBuilder {

    public static final int DEFAULT_SAMPLING_SIZE = 256;
    public static final int DEFAULT_NUM_TREES = 100;

    private double[][] data;

    public IsolationForestBuilder(double[][] data) {
        this.data = data;
    }

    private double[][] filterBy (double[][] data, int column, Function<Double, Boolean> predicate) {
        int n = data.length;
        int m = data[0].length;
        ArrayList<Integer> rowsIndices = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (predicate.apply(data[i][column])) {
                rowsIndices.add(i);
            }
        }
        double[][] filtered = new double[rowsIndices.size()][m];
        for (int i = 0; i < filtered.length; i++) {
            int idx = rowsIndices.get(i);
            filtered[i] = data[idx];
        }
        return filtered;
    }

    private int getRandomFeature (double[][] data) {
        Random random = new Random();
        return random.nextInt(data[0].length);
    }

    private double[] getFeature (double[][] data, int feature) {
        int n = data.length;
        double[] column = new double[n];
        for (int i = 0; i < n; i++) {
            column[i] = data[i][feature];
        }
        return column;
    }

    private double getRandomSplitValue (double[][] data, int feature) {
        double[] column = getFeature(data, feature);
        double max = Arrays.stream(column).max().orElse(0.0);
        double min = Arrays.stream(column).min().orElse(Double.MIN_VALUE);
        return (max - min)*Math.random() + min;
    }

    private Pair<double[][], double[][]> splitByRandomFeature (double[][] data, int feature, double splitValue) {
        double[][] lower = filterBy(data, feature, x -> x <= splitValue);
        double[][] higher = filterBy(data, feature, x -> x > splitValue);
        return new Pair<>(lower, higher);
    }

    private IsolationTree buildIsolationTree (double[][] data, double currentHeight, double maxHeight) {
        int size = data.length * (data.length == 0 ? 0 : data[0].length);
        if (currentHeight >= maxHeight || size <= 1) {
            return new IsolationLeaf(size);
        } else {
            int feature = getRandomFeature(data);
            double splitValue = getRandomSplitValue(data, feature);
            Pair<double[][], double[][]> splits = splitByRandomFeature(data, feature, splitValue);
            double[][] dataLeft = splits.first;
            double[][] dataRight = splits.second;
            return new IsolationTree(
                    feature,
                    splitValue,
                    buildIsolationTree(dataLeft, currentHeight + 1, maxHeight),
                    buildIsolationTree(dataRight, currentHeight + 1, maxHeight)
            );
        }
    }

    private double[][] sample (double[][] data, int samplingSize) {
        Random random = new Random();
        int n = data.length;
        int m = data[0].length;
        int dataSampleSize = Math.min(samplingSize, n);
        double[][] dataSample = new double[dataSampleSize][m];
        boolean[] taken = new boolean[n];
        for (int i = 0; i < dataSampleSize; i++) {
            int idx = random.nextInt(n);
            while (taken[idx]) {
                idx = random.nextInt(n);
            }
            taken[idx] = true;
            dataSample[i] = data[idx];
        }
        return dataSample;
    }

    public IsolationForest buildForest (int numTrees, int samplingSize) {
        ArrayList<IsolationTree> forest = new ArrayList<>();
        int maxHeight = (int) Math.ceil(Math.log(numTrees) / Math.log(2) + 1e-11);
        for (int i = 0; i < numTrees; i++) {
            double[][] sampledData = sample(this.data, samplingSize);
            IsolationTree tree = buildIsolationTree(sampledData, 0, maxHeight);
            forest.add(tree);
        }
        return new IsolationForest(forest);
    }

    public static void main(String[] args) throws Exception{
        String filepath = "./data/multivariateNormal.csv";
        BufferedReader br = new BufferedReader(new FileReader(filepath));
        String line = "";
        ArrayList<double[]> data = new ArrayList<>();
        // Skip column id's
        br.readLine();
        while ((line = br.readLine()) != null) {
            String[] nums = line.split(",");
            int m = nums.length;
            double[] row = new double[m - 1];
            for (int i = 0; i < m - 1; i++) {
                row[i] = Double.parseDouble(nums[i + 1]);
            }
            data.add(row);
        }

        int n = data.size();
        int m = data.get(0).length;
        double[][] dataMatrix = new double[n][m];
        for (int i = 0; i < n; i++) {
            dataMatrix[i] = data.get(i);
        }

        IsolationForestBuilder forestBuilder = new IsolationForestBuilder(dataMatrix);
        IsolationForest forest = forestBuilder.buildForest(
                20,
                1560
        );

        List<AnomalyClassification> classifications = forest.classifyData(
                dataMatrix,
                IsolationForestBuilder.DEFAULT_SAMPLING_SIZE
        );
        for (int i = 0; i < classifications.size(); i++) {
            System.out.println("i = " + i + " . classification: " + classifications.get(i).toString());
        }
    }
}
