package algorithm.gradient.descent;

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

    private List<List<Double>> list = new ArrayList<>();

    @Before
    public void createList() {
        List<Double> listA = new ArrayList<>();
        listA.add(1.0);
        listA.add(2.0);
        listA.add(Math.E);
        listA.add(10.0);
        list.add(listA);

        List<Double> listB = new ArrayList<>();
        listB.add(5.0);
        listB.add(8.0);
        listB.add(50.0);
        listB.add(30.0);
        list.add(listB);
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
        Matrix matrix = baseAlgorithm.getParamsMatrix(list);
        System.out.println(matrix.power(Calculation.Ret.NEW, 2.0));
        System.out.println(matrix.selectColumns(Calculation.Ret.NEW, 0));
    }

    @Test
    @Ignore
    public void couldStopStudy() {
    }
}