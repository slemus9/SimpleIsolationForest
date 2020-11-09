import java.util.*;
import java.util.stream.Collectors;

/**
 * Implementation of Isolation Forest based on the document:
 * https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest
 */
public class IsolationForest {

    private static final double H_CONSTANT = 0.5772156649;

    private static final double ANOMALY_THRESHOLD = 0.6;

    private List<IsolationTree> forest;

    public IsolationForest (List<IsolationTree> forest) {
        this.forest = forest;
    }

    private double harmonicNumber (double x) {
        return Math.log(x) + H_CONSTANT;
    }

    private double averagePathLength (int n) {
        return 2*harmonicNumber(n - 1) - 2*(n - 1.0)/n;
    }

    private double pathLenght (double[] instance, IsolationTree tree, int currentPathLength) {
        if (tree instanceof IsolationLeaf) {
            int currentSize = ((IsolationLeaf) tree).size;
            double adjustment = currentSize == 0 ? 0 : averagePathLength(currentSize);
            return currentPathLength + adjustment;
        } else {
            int feature = tree.splitFeature;
            double featureValue = instance[feature];
            if (featureValue < tree.splitValue) {
                return pathLenght(instance, tree.left, currentPathLength + 1);
            } else {
                return pathLenght(instance, tree.right, currentPathLength + 1);
            }
        }
    }

    private double expectedPathLenght (double[] instance) {
        double pathSum = 0;
        for (IsolationTree tree : forest) {
            double p = pathLenght(instance, tree, 0);
            pathSum += p;
        }
        return pathSum / forest.size();
    }

    public double anomalyScore (double[] instance, int n) {
        return Math.pow(2, -expectedPathLenght(instance)/averagePathLength(n));
    }

    public List<AnomalyClassification> classifyData (double[][] data, int samplingSize) {
        ArrayList<Double> scores = new ArrayList<>();
        for (double[] instance : data) {
            scores.add(anomalyScore(instance, samplingSize));
        }

//        for (int i = 0; i < scores.size(); i++) {
//            double ap = expectedPathLenght(data[i]);
//            if (ap < 10)
//                System.out.println("i = " + i + ". score = " + scores.get(i) + ". average path lenght: " + ap);
//        }

        return scores.stream().map(score -> {
            if (score >= ANOMALY_THRESHOLD) return AnomalyClassification.ANOMALY;
            else return AnomalyClassification.NORMAL;
        }).collect(Collectors.toList());
    }
}
