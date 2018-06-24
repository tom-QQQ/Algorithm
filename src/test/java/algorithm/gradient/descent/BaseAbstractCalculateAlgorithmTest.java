package algorithm.gradient.descent;

import algorithm.gradient.descent.LogisticRegression;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/23
 */
public class BaseAbstractCalculateAlgorithmTest {

    private List<Double> list;

    @Before
    public void createList() {
        list = new ArrayList<>();
        list.add(1.0);
        list.add(2.0);
        list.add(Math.E);
        list.add(10.0);
    }

    @Test
    @Ignore
    public void calculateCostWithMatrix() {
    }

    @Test
    @Ignore
    public void calculateHypothesisMatrix() {
    }

    @Test
    @Ignore
    public void calculateRegressionResultByMatrixWithGradientDescent() {
    }

    @Test
    @Ignore
    public void normalListValue() {
    }

    @Test
    @Ignore
    public void calculateNormalizationData() {
    }

    @Test
    @Ignore
    public void initCoefficientList() {
    }

    @Test
    @Ignore
    public void getParamsMatrix() {
    }

    @Test
    public void createMatrixWithList() {
        LogisticRegression baseAlgorithm = new LogisticRegression();
        Matrix matrix = baseAlgorithm.createMatrixWithList(list);
        matrix.log(Calculation.Ret.ORIG);
        System.out.println(matrix);
    }

    @Test
    @Ignore
    public void couldStopStudy() {
    }
}