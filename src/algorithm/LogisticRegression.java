package algorithm;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/20
 */
public class LogisticRegression extends BaseAlgorithm {

    public void calculateExampleResult() {

        List<List<Double>> dataParamsList = new ArrayList<>();


        List<Double> dataResults = new ArrayList<>();


        Matrix result = calculateRegressionResultByMatrixWithGradientDescent(dataParamsList, dataResults);
        System.out.println(result);
    }


    /**
     * 计算逻辑回归代价，公式较为复杂，这里不给出
     * @param paramsMatrix 数据矩阵
     * @param resultsMatrix 结果矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 逻辑回归代价
     */
    @Override
    Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {
        return null;
    }

    /**
     * 计算假设值矩阵，公式：hθ(x) = 1/(1-e^-(x*theta))
     * @param paramsMatrix 数据矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 假设值
     */
    @Override
    Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix) {
        return null;
    }
}
