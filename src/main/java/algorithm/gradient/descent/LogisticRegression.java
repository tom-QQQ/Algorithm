package algorithm.gradient.descent;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;
import utils.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/20
 */
public class LogisticRegression extends BaseAbstractCalculateAlgorithm {

    public LogisticRegression(boolean ifNeedSquare, boolean ifNeedTwoParamMultiply) {

        super(ifNeedSquare, ifNeedTwoParamMultiply);
    }

    public void calculateExampleResult() {

        List<List<Double>> dataParamsList = new ArrayList<>();

        addDataToDataParamsList(dataParamsList, 50.0, 45.0);
        addDataToDataParamsList(dataParamsList, 20.0, 60.0);
        addDataToDataParamsList(dataParamsList, 100.0, 20.0);
        addDataToDataParamsList(dataParamsList, 150.0, 20.0);
        addDataToDataParamsList(dataParamsList, 200.0, 20.0);
        addDataToDataParamsList(dataParamsList, 80.0, 10.0);

        List<Double> dataResults = new ArrayList<>();
        dataResults.add(0.0);
        dataResults.add(0.0);
        dataResults.add(1.0);
        dataResults.add(1.0);
        dataResults.add(1.0);
        dataResults.add(0.0);

        calculateRegressionResultByMatrixWithGradientDescent(dataParamsList, dataResults, true);
    }

    public void verificationResult() {

        List<Double> coefficient = new ArrayList<>();
        coefficient.add(2.5262);
        coefficient.add(9.2226);
        coefficient.add(1.4739);

        List<Double> dataParams = new ArrayList<>();
        dataParams.add(1.0);
        dataParams.add(80.0);
        dataParams.add(10.0);

        normalListValue(dataParams);
        double result = calculateHypothesisResult(dataParams, coefficient);
        result = Functions.sigmoid(result);
        dataParams.clear();
        System.out.println("逻辑回归结果：" + result);
    }


    /**
     * 计算逻辑回归代价，公式较为复杂，这里不给出，这里的处理方式为：当结果为0时，
     * 假设值修改为1减去之前的假设值；结果为1时，假设值为本身，然后求出矩阵log值，计算全部值的和，除以数据条数后取相反数,再加正则部分的代价λ∑θ^2，
     *
     * @param paramsMatrix 数据矩阵
     * @param resultsMatrix 结果矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 逻辑回归代价
     */
    @Override
    Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        Matrix hypothesisMatrix = calculateHypothesisMatrix(paramsMatrix, coefficientMatrix);
        long rowCount = hypothesisMatrix.getRowCount();

        changeValueForCalculateLog(hypothesisMatrix, resultsMatrix, rowCount);

        hypothesisMatrix.log(Calculation.Ret.ORIG);

        return calculateCostWithRegular(hypothesisMatrix, coefficientMatrix, rowCount);

    }

    /**
     * 计算假设值矩阵，公式：hθ(x) = 1/(1-e^-(x*theta))，结果∈[0,1]
     * @param paramsMatrix 数据矩阵 m*n
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 假设值
     */
    @Override
    Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix) {

        Matrix resultMatrix = paramsMatrix.mtimes(coefficientMatrix);
        resultMatrix.logistic(Calculation.Ret.ORIG);

        return resultMatrix;
    }

    /**
     * 为逻辑回归的log值计算修正参数
     * @param hypothesisMatrix 假设结果矩阵 m*1
     * @param resultsMatrix 结果矩阵 m*!
     * @param rowCount m
     */
    private void changeValueForCalculateLog(Matrix hypothesisMatrix, Matrix resultsMatrix, long rowCount) {

        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            double resultValue = resultsMatrix.getAsDouble(rowIndex, 0);
            if (resultValue == 0.0) {
                double hypothesisValue = hypothesisMatrix.getAsDouble(rowIndex, 0);
                hypothesisMatrix.setAsDouble(1 - hypothesisValue, rowIndex, 0);
            }
        }
    }

    /**
     * 计算正则化代价：-1/m(假设值的log结果和 - 正则代价/2)
     * @param hypothesisMatrix 假设结果矩阵 m*1
     * @param coefficientMatrix 系数矩阵 n*1
     * @param rowCount 数据数量
     * @return 正则代价
     */
    private double calculateCostWithRegular(Matrix hypothesisMatrix, Matrix coefficientMatrix, long rowCount) {

        double costValue = hypothesisMatrix.getValueSum();
        double regularPartCost = calculateRegularPartCostValue(coefficientMatrix)/2;
        return - (costValue - regularPartCost)/rowCount;
    }
}
