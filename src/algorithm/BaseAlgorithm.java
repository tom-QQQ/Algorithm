package algorithm;


import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/19
 */
 abstract class BaseAlgorithm {



    private int initialNumberRange;
    double studyRate;
    double declineValue;
    double convergence;
    long maxLoop;

    {
        studyRate = 1.0;
        declineValue = 0.618;
        convergence = 0.0001;
        maxLoop = 1000L;
        initialNumberRange = 100;
    }



    List<Double> averageValues = new ArrayList<>();
    List<Double> squaredDifferenceValues = new ArrayList<>();


    /**
     * 计算代价
     * @param paramsMatrix 数据矩阵
     * @param resultsMatrix 结果矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 代价
     */
    abstract Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix);

    /**
     * 计算假设值矩阵，hθ(x)
     * @param paramsMatrix 数据矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 新的系数矩阵
     */
    abstract Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix);

    /**
     * 借助矩阵使用梯度下降法计算系数结果
     * @param dataParamsList 数据list
     * @param dataResults 结果list
     * @return 结果系数矩阵
     */
    Matrix calculateRegressionResultByMatrixWithGradientDescent(List<List<Double>> dataParamsList, List<Double> dataResults) {

        if (dataParamsList.size() != dataResults.size()) {
            return null;
        }

        List<List<Double>> normalizationDataList = calculateNormalizationData(dataParamsList);

        if (normalizationDataList == null) {
            return null;
        }

        Matrix paramsMatrix = getParamsMatrix(normalizationDataList);
        Matrix resultsMatrix = createMatrixWithList(dataResults);

        List<Double> coefficientList = initCoefficientList(dataParamsList.get(0).size() + 1);
        Matrix coefficientMatrix = createMatrixWithList(coefficientList);

        return calculateCoefficientWithIterative(paramsMatrix, resultsMatrix, coefficientMatrix);

    }

    /**
     * 规格化检验数据，应在调用回归方法计算数据的平均值和方差后调用
     * @param dataParams 数据
     * @param averageValues 数据平均值
     * @param squaredDifferenceValues 数据方差
     */
    void normalListValue(List<Double> dataParams, List<Double> averageValues, List<Double> squaredDifferenceValues) {

        for (int valueIndex = 1; valueIndex < dataParams.size(); valueIndex++) {
            double value = dataParams.get(valueIndex);
            dataParams.set(valueIndex, (value - averageValues.get(valueIndex-1))/squaredDifferenceValues.get(valueIndex-1));
        }
    }

    /**
     * 对原始数据进行规格化操作，并在每列数据添加一个1
     * @param dataParamsList 原始数据
     * @return 规格化结果
     */
    List<List<Double>> calculateNormalizationData(List<List<Double>> dataParamsList) {

        List<Double> averageValues = calculateAverageValueList(dataParamsList);
        this.averageValues.addAll(averageValues);

        List<Double> squaredDifferenceList = calculateStandardDeviationList(dataParamsList, averageValues);
        this.squaredDifferenceValues.addAll(squaredDifferenceList);

        if (averageValues.size() != squaredDifferenceList.size() || averageValues.size() != dataParamsList.size()) {
            return null;
        }

        return getNormalizationParamsList(dataParamsList, averageValues, squaredDifferenceList);
    }


    /**
     * 随机初始化结果系数的值
     * @param coefficientsSize 系数个数
     * @return 初始结果系数List
     */
    List<Double> initCoefficientList(int coefficientsSize) {

        List<Double> coefficientList = new ArrayList<>(coefficientsSize);

        for (int index = 0; index < coefficientsSize; index++) {
            coefficientList.add(createInitNumberInRange());
        }

        return coefficientList;
    }

    /**
     * 根据规格化完成的数据创建矩阵
     * @param dataParamsList 规格化完毕的数据
     * @return 对应矩阵
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
     * 根据结果数据创建对应矩阵
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
     * 迭代计算系数结果
     * @param paramsMatrix 数据矩阵
     * @param resultsMatrix 结果矩阵
     * @param coefficientMatrix 系数矩阵
     * @return 最终迭代结果
     */
    private Matrix calculateCoefficientWithIterative(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        int calculateTimes = 0;

        while (true) {

            double previousCostValue = calculateCostWithMatrix(paramsMatrix, resultsMatrix, coefficientMatrix);

            Matrix hypothesisMatrix = calculateHypothesisMatrix(paramsMatrix, coefficientMatrix);

            Matrix newCoefficientMatrix = calculateDifferenceCoefficientMatrix(hypothesisMatrix, paramsMatrix, resultsMatrix, coefficientMatrix);

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

    /**
     * 计算新参数和之前参数的差值，公式： ɑ/m[(hθ(x)-y)^T*x]^T
     * @param hypothesisMatrix 假设结果矩阵
     * @param paramsMatrix 参数矩阵
     * @param resultsMatrix 结果矩阵
     * @return 新参数和之前参数的差值
     */
    private Matrix calculateDifferenceCoefficientMatrix(Matrix hypothesisMatrix, Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        Matrix differenceCoefficientMatrix = hypothesisMatrix.minus(resultsMatrix);
        differenceCoefficientMatrix = differenceCoefficientMatrix.transpose().mtimes(paramsMatrix).transpose();
        differenceCoefficientMatrix =  differenceCoefficientMatrix.times(studyRate).divide(resultsMatrix.getRowCount());

        return coefficientMatrix.minus(differenceCoefficientMatrix);

    }

    boolean couldStopStudy(double previousCostValue, double currentCostValue, double costMinValue) {

        double difference = previousCostValue - currentCostValue;

        return difference < costMinValue;
    }

    /**
     * 生成指定绝对值范围内的随机值
     * @return 指定范围的随机值
     */
    private double createInitNumberInRange() {

        double number = Math.random()* initialNumberRange;

        boolean ifNegative = Math.random() < 0.5;

        return (ifNegative ? -number : number);
    }

    /**
     * 计算给定List<>中List的平均值
     * @param dataParams 需要计算的数据
     * @return 平均值list
     */
    private List<Double> calculateAverageValueList(List<List<Double>> dataParams) {

        List<Double> averageValues = new ArrayList<>(dataParams.get(0).size());

        for (int valueIndex = 0; valueIndex < dataParams.get(0).size(); valueIndex++) {

            Double sumValues = 0.0;
            for (List<Double> valueList : dataParams) {
                sumValues += valueList.get(valueIndex);
            }

            averageValues.add(sumValues/dataParams.size());
        }

        return averageValues;
    }

    /**
     * 计算给定List<>中List的方差
     * @param dataParams 需要计算的数据
     * @param averageValues 数据的平局数
     * @return 方差list
     */
    private List<Double> calculateStandardDeviationList(List<List<Double>> dataParams, List<Double> averageValues) {

        List<Double> standardDeviationList = new ArrayList<>(averageValues.size());

        for (int valueIndex = 0; valueIndex < averageValues.size(); valueIndex++) {

            Double standardDeviation = 0.0;
            for (List<Double> valueList : dataParams) {
                standardDeviation += calculateDeviationSquare(valueList.get(valueIndex), averageValues.get(valueIndex));
            }

            standardDeviationList.add(Math.sqrt(standardDeviation/dataParams.size()));
        }
        return standardDeviationList;
    }

    /**
     * 计算两个值的平方差
     * @param valueOne 第一个值
     * @param valueTwo 第二个值
     * @return 平方差
     */
    private double calculateDeviationSquare(double valueOne, double valueTwo) {

        return (valueOne - valueTwo)*(valueOne - valueTwo);
    }

    /**
     * 规格化数据，规格方法: (value - 平均值)/方差
     * @param dataValues 要规格化的数据
     * @param averageValues 数据的平均值
     * @param squaredDifferenceList 数据的方差
     * @return 规格化结果,每个list的第一项值为1
     */
    private List<List<Double>> getNormalizationParamsList(List<List<Double>> dataValues, List<Double> averageValues, List<Double> squaredDifferenceList) {

        List<List<Double>> normalizationResults = initNormalizationResultsList(dataValues.size(), dataValues.get(0).size());

        for (int valueIndex = 0; valueIndex < averageValues.size(); valueIndex++) {

            for (int listIndex = 0; listIndex < dataValues.size(); listIndex++) {

                Double normalizationValue = calculateNormalizationValue(dataValues.get(listIndex).get(valueIndex), averageValues.get(valueIndex), squaredDifferenceList.get(valueIndex));
                normalizationResults.get(listIndex).add(normalizationValue);
            }
        }

        return normalizationResults;
    }

    /**
     * 创建一个新的数据list，并将第一个值初始化为1
     * @param listSize 参数条数
     * @param valueSize 参数个数
     * @return list
     */
    private List<List<Double>> initNormalizationResultsList(int listSize, int valueSize) {

        List<List<Double>> normalizationResultsList = new ArrayList<>(listSize);

        for (int listIndex = 0; listIndex < listSize; listIndex++) {

            List<Double> normalizationResults = new ArrayList<>(valueSize + 1);
            normalizationResults.add(1.0);
            normalizationResultsList.add(normalizationResults);
        }

        return normalizationResultsList;
    }

    /**
     * 计算规格化值
     * @param value 初始值
     * @param average 平均值
     * @param squaredDifference 方差
     * @return 规格化结果
     */
    private Double calculateNormalizationValue(Double value, Double average, Double squaredDifference) {

        return (value - average)/squaredDifference;
    }
}
