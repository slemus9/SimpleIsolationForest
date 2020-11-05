import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Implementation of Isolation Forest based on the document:
 * https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest
 */
public class IsolationForest {

    private double[][] data;
    private double[][] dataTransposed;
    private String[] features;
    private Map<String, Integer> featuresIdx = new HashMap<>();

    public IsolationForest(double[][] data, String[] features) {
        this.data = data;
        this.features = features;
        initDataTransposed(data);
        initFeatureIndices(features);
    }

    private void initFeatureIndices (String[] features) {
        for (int i = 0; i < features.length; i++) {
            featuresIdx.put(features[i], i);
        }
    }

    private void initDataTransposed (double[][] data) {
        int n = data.length;
        int m = data[0].length;
        dataTransposed = new double[m][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                dataTransposed[j][i] = data[i][j];
            }
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
}
