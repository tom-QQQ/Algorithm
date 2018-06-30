package algorithm.gradient.descent;


import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

import java.math.BigDecimal;
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
    /**
     * 正则系数，为0时不进行正则化,当正则系数较小但仍然大于0时，需要迭代计算更多次，但更容易发现相关性较小的参数，但再向0接近时，迭代次数又会减少
     * 如果正则系数较大，则代价也会增加很多，推荐范围[0.005, 1]
     */
    private double lambda;

    /**
     * 初始范围和结果范围警告倍数
     */
    private int referenceInitAndActualTimes = 5;

    {
        initialNumberRange = 10;
        studyRate = 1.0;
        declineValue = 0.618;
        maxLoop = 1000L;
        convergence = 0.0001;
        lambda = 1;
    }

    BaseAbstractCalculateAlgorithm(boolean ifNeedSquare, boolean ifNeedTwoParamMultiply) {

        super(ifNeedSquare, ifNeedTwoParamMultiply);
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
     */
    void calculateRegressionResultByMatrixWithGradientDescent(List<List<Double>> dataParamsList, List<Double> dataResults, boolean ifNeedCalculateResult) {

        if (dataParamsList.size() != dataResults.size()) {
            System.out.println("参数list数量和结果数量不同");
            return;
        }

        List<List<Double>> normalizationDataList = calculateNormalizationData(dataParamsList);

        if (normalizationDataList == null) {
            System.out.println("规格化结果为空");
            return;
        }

        if (!ifNeedCalculateResult) {
            System.out.println("正在进行参数验证");
            return;
        }

        Matrix paramsMatrix = getParamsMatrix(normalizationDataList);
        Matrix resultsMatrix = createMatrixWithList(dataResults);

        List<Double> coefficientList = initCoefficientList(dataParamsList.get(0).size() + 1);
        Matrix coefficientMatrix = createMatrixWithList(coefficientList);

        Matrix resultMatrix =  calculateCoefficientWithIterative(paramsMatrix, resultsMatrix, coefficientMatrix);
        System.out.println(resultMatrix);

        referenceInitCoefficientRange(resultMatrix);
    }

    /**
     * 计算代价中的正则部分
     * @param coefficientMatrix 结果矩阵
     * @return 正则代价部分
     */
    double calculateRegularPartCostValue(Matrix coefficientMatrix) {

        double theta0 = coefficientMatrix.getAsDouble(0, 0);
        coefficientMatrix.setAsDouble(0.0, 0, 0);
        double sumThetasSquare = coefficientMatrix.power(Calculation.Ret.NEW, 2.0).getValueSum();
        coefficientMatrix.setAsDouble(theta0, 0, 0);
        return sumThetasSquare*lambda;
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

                if (couldStopStudy(previousCostValue, currentCostValue)) {
                    double realCost = realCost(newCoefficientMatrix, currentCostValue, resultsMatrix.getRowCount());
                    System.out.println("计算了" + calculateTimes + "次，迭代达到目标精度，迭代停止。 最终去除正则部分代价：" + realCost);
                    return newCoefficientMatrix;
                }

                if (calculateTimes == maxLoop) {
                    double realCost = realCost(newCoefficientMatrix, currentCostValue, resultsMatrix.getRowCount());
                    System.out.println("达到最大迭代次数" + maxLoop + "，迭代停止。 最终去除正则部分代价：" + realCost);
                    return newCoefficientMatrix;
                }
            }

            calculateTimes++;
            coefficientMatrix = newCoefficientMatrix;
        }
    }

    /**
     * 判断是否需要停止迭代，该方法无需在子类中调用
     * @param previousCostValue 之前的代价
     * @param currentCostValue 当前代价
     * @return 是否需要停止迭代
     */
    boolean couldStopStudy(double previousCostValue, double currentCostValue) {

        double difference = previousCostValue - currentCostValue;

        return difference < convergence;
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
     * 计算新参数和之前参数的差值，公式： θ(1 - ɑ*(λ/m)) - ɑ/m[(hθ(x) - y)^T*x]^T，计算θ第一项时，λ为0
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

    /**
     * 去除正则部分求出实际的代价
     * @param coefficientMatrix 结果矩阵
     * @param costValue 正则代价
     * @return 减去正则代价的代价
     */
    private double realCost(Matrix coefficientMatrix, double costValue, long rowCount) {

        double sumThetasSquare = calculateRegularPartCostValue(coefficientMatrix);
        return costValue - sumThetasSquare/2/rowCount;
    }


    /**
     * 打印建议初始值范围，除去最大数的平均值
     * @param resultMatrix 系数结果矩阵
     */
    private void referenceInitCoefficientRange(Matrix resultMatrix) {

        double maxValue = resultMatrix.getMaxValue();
        double sumValue = resultMatrix.getAbsoluteValueSum();
        double referenceValue = (sumValue - maxValue)/(resultMatrix.getRowCount() - 1);
        referenceValue = BigDecimal.valueOf(referenceValue).setScale(0, BigDecimal.ROUND_HALF_UP).intValue();

        if (Math.max(referenceValue, initialNumberRange)/Math.min(referenceValue, initialNumberRange) > referenceInitAndActualTimes) {
            System.out.println("根据系数结果，推荐初始结果系数范围最大绝对值为: " + referenceValue);
        }
    }
}
