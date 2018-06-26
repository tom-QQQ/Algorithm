package algorithm.gradient.descent;


import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/19
 */
 abstract class BaseAbstractCalculateAlgorithm extends BaseDataConstruct {

    double studyRate;
    double declineValue;
    long maxLoop;
    double convergence;
    double lambda;

    {
        initialNumberRange = 100;
        studyRate = 1.0;
        declineValue = 0.618;
        maxLoop = 1000L;
        convergence = 0.0001;
        lambda = 1.0;
    }

    /**
     * 计算代价,Jθ
     * @param paramsMatrix 数据矩阵 m*n
     * @param resultsMatrix 结果矩阵 m*1
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 代价
     */
    abstract Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix);

    /**
     * 计算假设值矩阵，hθ(x)
     * @param paramsMatrix 数据矩阵 m*n
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 新的系数矩阵
     */
    abstract Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix);

    /**
     * 借助矩阵使用梯度下降法计算系数结果
     * @param dataParamsList 数据list
     * @param dataResults 结果list
     * @return 结果系数矩阵
     */
    Matrix calculateRegressionResultByMatrixWithGradientDescent(List<List<Double>> dataParamsList, List<Double> dataResults, boolean ifNeedCalculateResult) {

        if (dataParamsList.size() != dataResults.size()) {
            System.out.println("参数list数量和结果数量不同");
            return null;
        }

        List<List<Double>> normalizationDataList = calculateNormalizationData(dataParamsList);

        if (normalizationDataList == null || !ifNeedCalculateResult) {
            return null;
        }

        Matrix paramsMatrix = getParamsMatrix(normalizationDataList);
        Matrix resultsMatrix = createMatrixWithList(dataResults);

        List<Double> coefficientList = initCoefficientList(dataParamsList.get(0).size() + 1);
        Matrix coefficientMatrix = createMatrixWithList(coefficientList);

        return calculateCoefficientWithIterative(paramsMatrix, resultsMatrix, coefficientMatrix);
    }

    /**
     * 迭代计算系数结果
     * @param paramsMatrix 数据矩阵 m*n
     * @param resultsMatrix 结果矩阵 m*1
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 最终迭代结果
     */
    private Matrix calculateCoefficientWithIterative(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        int calculateTimes = 0;

        while (true) {

            double previousCostValue = calculateCostWithMatrix(paramsMatrix, resultsMatrix, coefficientMatrix);

            Matrix hypothesisMatrix = calculateHypothesisMatrix(paramsMatrix, coefficientMatrix);

            Matrix newCoefficientMatrix = calculateNewCoefficientMatrix(hypothesisMatrix, paramsMatrix, resultsMatrix, coefficientMatrix);

            double currentCostValue = calculateCostWithMatrix(paramsMatrix, resultsMatrix, newCoefficientMatrix);

            if (currentCostValue > previousCostValue) {
                newCoefficientMatrix = coefficientMatrix;
                studyRate *= declineValue;

            } else {

                if (couldStopStudy(previousCostValue, currentCostValue, convergence)) {
                    System.out.println("计算了" + calculateTimes + "次，迭代达到目标精度，迭代停止。 最终代价：" + currentCostValue);
                    return newCoefficientMatrix;
                }

                if (calculateTimes == maxLoop) {
                    System.out.println("达到最大迭代次数" + maxLoop + "，迭代停止。 最终代价" + currentCostValue);
                    return newCoefficientMatrix;
                }
            }

            calculateTimes++;
            coefficientMatrix = newCoefficientMatrix;
        }
    }

    boolean couldStopStudy(double previousCostValue, double currentCostValue, double costMinValue) {

        double difference = previousCostValue - currentCostValue;

        return difference < costMinValue;
    }

    /**
     * 根据完成规格化的数据创建矩阵，该方法无需在子类中调用
     * @param dataParamsList 规格化完毕的数据
     * @return 对应矩阵 m*n
     */
    Matrix getParamsMatrix(List<List<Double>> dataParamsList) {

        int listSize = dataParamsList.get(0).size();

        Matrix matrix = Matrix.Factory.zeros(dataParamsList.size(), listSize);

        for (int listIndex = 0; listIndex < dataParamsList.size(); listIndex++) {

            for (int valueIndex = 0; valueIndex < listSize; valueIndex++) {
                matrix.setAsDouble(dataParamsList.get(listIndex).get(valueIndex), listIndex, valueIndex);
            }
        }

        return matrix;
    }

    /**
     * 根据结果数据创建对应矩阵，该方法无需在子类中调用
     * @param dataResults 结果数据
     * @return 对应矩阵
     */
    Matrix createMatrixWithList(List<Double> dataResults) {

        int resultSize = dataResults.size();

        Matrix matrix = Matrix.Factory.zeros(resultSize, 1);

        for (int valueIndex = 0; valueIndex < resultSize; valueIndex++) {
            matrix.setAsDouble(dataResults.get(valueIndex), valueIndex, 0);
        }

        return matrix;
    }

    /**
     * 计算新参数和之前参数的差值，公式： θ(1 - ɑ*(λ/m)) - ɑ/m[(hθ(x) - y)^T*x]^T，除θ第一项
     * @param hypothesisMatrix 假设结果矩阵 m*1
     * @param paramsMatrix 参数矩阵 m*n
     * @param resultsMatrix 结果矩阵 m*1
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 新系数矩阵 n*1
     */
    private Matrix calculateNewCoefficientMatrix(Matrix hypothesisMatrix, Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        double previousTheta0Value = coefficientMatrix.getAsDouble(0, 0);
        Matrix differenceCoefficientMatrix = calculateDifferenceCoefficientMatrix(hypothesisMatrix, paramsMatrix, resultsMatrix);

        regularCoefficientMatrix(coefficientMatrix);

        Matrix newCoefficientMatrix = coefficientMatrix.minus(differenceCoefficientMatrix);
        changeTheta0ToRightValue(previousTheta0Value, differenceCoefficientMatrix, newCoefficientMatrix);
        return newCoefficientMatrix;

    }

    /**
     * 计算需要正则化系数值需要减去的值矩阵
     * @param hypothesisMatrix 假设结果矩阵 m*1
     * @param paramsMatrix 参数矩阵 m*n
     * @param resultsMatrix 结果矩阵 m*1
     * @return 需要减去的值的矩阵 n*1
     */
    private Matrix calculateDifferenceCoefficientMatrix(Matrix hypothesisMatrix, Matrix paramsMatrix, Matrix resultsMatrix) {

        Matrix differenceCoefficientMatrix = hypothesisMatrix.minus(resultsMatrix);
        differenceCoefficientMatrix = differenceCoefficientMatrix.transpose().mtimes(paramsMatrix).transpose();
        return differenceCoefficientMatrix.times(studyRate).divide(resultsMatrix.getRowCount());
    }

    /**
     * “正则化”θ矩阵
     * @param coefficientMatrix 正则化结果 n*1
     */
    private void regularCoefficientMatrix(Matrix coefficientMatrix) {

        long size = coefficientMatrix.getRowCount();
        for (int rowIndex = 0; rowIndex < size; rowIndex++) {
            double value = coefficientMatrix.getAsDouble(rowIndex, 0);
            double regularResult = calculateRegularResult(value, size);
            coefficientMatrix.setAsDouble(regularResult, rowIndex, 0);
        }
    }

    /**
     * 计算“正则化”值
     * @param value 需要正则化的值
     * @return 正则化结果
     */
    private double calculateRegularResult(double value, long size) {
        return value*(1 - studyRate*lambda/size);
    }

    /**
     * 修正新的theta0的值
     * @param previousTheta0Value 之前theta0的值
     * @param differenceCoefficientMatrix 需要减去的值的矩阵 n*1
     * @param newCoefficientMatrix 需要修改的系数矩阵 n*1
     */
    private void changeTheta0ToRightValue(double previousTheta0Value, Matrix differenceCoefficientMatrix, Matrix newCoefficientMatrix) {

        double rightValue = previousTheta0Value - differenceCoefficientMatrix.getAsDouble(0, 0);
        newCoefficientMatrix.setAsDouble(rightValue, 0, 0);
    }
}
