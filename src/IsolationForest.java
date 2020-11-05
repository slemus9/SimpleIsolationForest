import java.util.*;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Implementation of Isolation Forest based on the document:
 * https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest
 */
public class IsolationForest {

    private double[][] data;
    private String[] features;
    private Map<String, Integer> featuresIdx = new HashMap<>();

    public IsolationForest(double[][] data, String[] features) {
        this.data = data;
        this.features = features;
        initFeatureIndices(features);
    }

    private void initFeatureIndices (String[] features) {
        for (int i = 0; i < features.length; i++) {
            featuresIdx.put(features[i], i);
        }
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

    private int getRandomFeature () {
        Random random = new Random();
        return random.nextInt(data[0].length);
    }

    private double getRandomSplitValue (double[][] data, int feature) {
        DoubleStream column = Arrays.stream(data[feature]);
        double max = column.max().orElse(0.0);
        double min = column.min().orElse(Double.MIN_VALUE);
        return (max - min)*Math.random() + min;
    }

    private Pair<double[][], double[][]> splitByRandomFeature (double[][] data, int feature, double splitValue) {
        double[][] lower = filterBy(data, feature, x -> x <= splitValue);
        double[][] higher = filterBy(data, feature, x -> x > splitValue);
        return new Pair<>(lower, higher);
    }

    private IsolationTree buildIsolationTree (double[][] data, double currentHeight, double maxHeight) {
        int size = data.length * data[0].length;
        if (currentHeight >= maxHeight || size <= 1) {
            return new IsolationLeaf(size);
        } else {
            int feature = getRandomFeature();
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
}
