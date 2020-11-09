public class IsolationTree {

    int splitFeature;
    double splitValue;
    IsolationTree left;
    IsolationTree right;

    public IsolationTree (
            int splitFeature,
            double splitValue,
            IsolationTree left,
            IsolationTree right) {
        this.splitFeature = splitFeature;
        this.splitValue = splitValue;
        this.left = left;
        this.right = right;
    }
}