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


        Matrix result = calculateRegressionResultByMatrixWithGradientDescent(dataParamsList, dataResults);
        System.out.println(result);
    }

    public void verificationResult() {

        List<Double> coefficient = new ArrayList<>();
        coefficient.add(12.5553);
        coefficient.add(27.0026);
        coefficient.add(7.1081);

        List<Double> dataParams = new ArrayList<>();
        dataParams.add(1.0);
        dataParams.add(0.0);
        dataParams.add(0.0);

        normalListValue(dataParams, averageValues, squaredDifferenceValues);
        double result = calculateHypothesisResult(dataParams, coefficient);
        result = Functions.sigmoid(result);
        dataParams.clear();
        System.out.println(result);
    }


    /**
     * 计算逻辑回归代价，公式较为复杂，这里不给出，这里的处理方式为：当结果为0时，
     * 假设值修改为1减去之前的假设值，结果为1时，假设值为本身，然后求出矩阵log值，计算全部值的和，再除以数据条数
     *
     * @param paramsMatrix 数据矩阵
     * @param resultsMatrix 结果矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 逻辑回归代价
     */
    @Override
    Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        Matrix hypothesisMatrix = calculateHypothesisMatrix(paramsMatrix, coefficientMatrix);

        for (int rowIndex = 0; rowIndex < hypothesisMatrix.getRowCount(); rowIndex++) {
            double resultValue = resultsMatrix.getAsDouble(rowIndex, 0);
            if (resultValue == 0.0) {
                double hypothesisValue = hypothesisMatrix.getAsDouble(rowIndex, 0);
                hypothesisMatrix.setAsDouble(1 - hypothesisValue, rowIndex, 0);
            }
        }

        hypothesisMatrix.log(Calculation.Ret.ORIG);
        return -hypothesisMatrix.getValueSum()/resultsMatrix.getRowCount();
    }

    /**
     * 计算假设值矩阵，公式：hθ(x) = 1/(1-e^-(x*theta))
     * @param paramsMatrix 数据矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 假设值
     */
    @Override
    Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix) {

        Matrix resultMatrix = paramsMatrix.mtimes(coefficientMatrix);
        resultMatrix.logistic(Calculation.Ret.ORIG);

        return resultMatrix;
    }
}
