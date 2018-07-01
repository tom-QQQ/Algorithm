package algorithm.gradient.descent;


import org.apache.commons.math3.linear.RealMatrix;
import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;


/**
 * @author Ning
 * @date Create in 2018/5/31
 */
public class LinearRegression extends BaseAbstractCalculateAlgorithm {

    /**
     * 设置是否需要平方项和两两相乘项
     * @param ifNeedSquare 是否需要平方项
     * @param ifNeedTwoParamMultiply 是否需要两两相乘项
     */
    public LinearRegression(boolean ifNeedSquare, boolean ifNeedTwoParamMultiply) {

        super(ifNeedSquare, ifNeedTwoParamMultiply);
    }

    public void calculateExampleResult() {

        List<List<Double>> dataParamsList = new ArrayList<>();

        addDataToDataParamsList(dataParamsList, 2104.0, 5.0, 1.0, 45.0);
        addDataToDataParamsList(dataParamsList, 1416.0, 3.0, 2.0, 40.0);
        addDataToDataParamsList(dataParamsList, 1534.0, 3.0, 2.0, 30.0);
        addDataToDataParamsList(dataParamsList, 852.0, 2.0, 1.0, 36.0);

        List<Double> dataResults = new ArrayList<>();
        dataResults.add(460.0);
        dataResults.add(232.0);
        dataResults.add(315.0);
        dataResults.add(178.0);

        // 循环回归
//        List<Double> results = calculateRegressionResultWithGradientDescent(dataParamsList, dataResults);
//        if (results!= null) {
//            System.out.println("回归结果：");
//            Utils.printList(results);
//        }

        // 矩阵回归
        calculateRegressionResultByMatrixWithGradientDescent(dataParamsList, dataResults, true);

//        RealMatrix realResult = calculateResult(dataParamsList, dataResults);
//        printMatrix(realResult);

//        Matrix result = calculateRegressionResultWithNormalEquation(dataParamsList, dataResults);
//        System.out.print(result);
    }

    /**
     * 检验结果
     */
    public void verificationResult() {

        List<Double> coefficient = new ArrayList<>();
        coefficient.add(296.2500);
        coefficient.add(62.5568);
        coefficient.add(56.9099);
        coefficient.add(-25.4038);
        coefficient.add(-32.8265);

        List<Double> dataParams = new ArrayList<>();
        dataParams.add(1.0);
        dataParams.add(2104.0);
        dataParams.add(5.0);
        dataParams.add(1.0);
        dataParams.add(45.0);
        normalListValue(dataParams);
        double result = calculateHypothesisResult(dataParams, coefficient);
        dataParams.clear();
        System.out.println(result);

        dataParams.clear();
        dataParams.add(1.0);
        dataParams.add(1416.0);
        dataParams.add(3.0);
        dataParams.add(2.0);
        dataParams.add(40.0);
        normalListValue(dataParams);
        result = calculateHypothesisResult(dataParams, coefficient);
        dataParams.clear();
        System.out.println(result);

        dataParams.clear();
        dataParams.add(1.0);
        dataParams.add(1534.0);
        dataParams.add(3.0);
        dataParams.add(2.0);
        dataParams.add(30.0);
        normalListValue(dataParams);
        result = calculateHypothesisResult(dataParams, coefficient);
        dataParams.clear();
        System.out.println(result);

        dataParams.clear();
        dataParams.add(1.0);
        dataParams.add(852.0);
        dataParams.add(2.0);
        dataParams.add(1.0);
        dataParams.add(36.0);
        normalListValue(dataParams);
        result = calculateHypothesisResult(dataParams, coefficient);
        dataParams.clear();dataParams.clear();
        System.out.println(result);

    }

    /**
     * 计算正则化代价, 公式: 1/(2*m)*[∑(x*θ-y)^2 + λ∑θ^2)] ，不正则化常数项
     * @param paramsMatrix 数据矩阵 m*n
     * @param resultsMatrix 结果矩阵 m*1
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 线性回归代价
     */
    @Override
    Double calculateCostWithMatrix(Matrix paramsMatrix, Matrix resultsMatrix, Matrix coefficientMatrix) {

        Matrix deviationMatrix = paramsMatrix.mtimes(coefficientMatrix).minus(resultsMatrix);

        double costResult = 0.0;

        for (int row = 0; row < deviationMatrix.getRowCount(); row++) {

            Double originalValue = deviationMatrix.getAsDouble(row, 0);
            costResult += originalValue * originalValue;
        }

        costResult += calculateRegularPartCostValue(coefficientMatrix);

        return costResult/resultsMatrix.getRowCount()/2;
    }


    /**
     * 计算假设值矩阵，公式：hθ(x) = x*theta
     * @param paramsMatrix 数据矩阵 m*n
     * @param coefficientMatrix 系数矩阵 n*1
     * @return 新的系数矩阵 n*1
     */
    @Override
    Matrix calculateHypothesisMatrix(Matrix paramsMatrix, Matrix coefficientMatrix) {

         return paramsMatrix.mtimes(coefficientMatrix);
    }


    // =================================以下为未使用矩阵未进行正则化进行的线性梯度下降=========================================

    /**
     * 使用梯度下降法计算线性方程参数
     * @param dataParamsList 参数集合，按照相同顺序组成的列表的集合,size()>1
     * @param dataResults 结果集合,一个参数列表的索引和对应结果在结果集合中的索引相同
     * @return 线性方程参数
     */
    private List<Double> calculateRegressionResultWithGradientDescent(List<List<Double>> dataParamsList, List<Double> dataResults) {

        if (dataParamsList.size() != dataResults.size()) {
            return null;
        }

        List<List<Double>> normalizationParamsList = calculateNormalizationData(dataParamsList);

        if (normalizationParamsList == null) {
            return null;
        }

        List<Double> coefficientList = initCoefficientList(dataParamsList.get(0).size() + 1);

        return calculateCoefficientListWithCycle(normalizationParamsList, dataResults,coefficientList);
    }

    /**
     * 迭代计算系数
     * @param normalizationParamsList 数据规格化结果
     * @param dataResults 数据结果值
     * @param coefficientList 结果系数
     * @return 结果系数
     */
    private List<Double> calculateCoefficientListWithCycle(List<List<Double>> normalizationParamsList, List<Double> dataResults, List<Double> coefficientList) {

        int calculateTimes = 0;
        while (true) {

            double previousCostValue = costFunctionResult(normalizationParamsList, dataResults, coefficientList);

            List<Double> newCoefficientList = calculateNewCoefficientValueList(coefficientList, normalizationParamsList, dataResults);

            double currentCostValue = costFunctionResult(normalizationParamsList, dataResults, newCoefficientList);

            if (currentCostValue > previousCostValue) {
                declineStudyRateAndRecoverCoefficient(newCoefficientList, coefficientList);

            } else {

                if (couldStopStudy(previousCostValue, currentCostValue)) {
                    System.out.println("计算了" + calculateTimes + "次，迭代达到目标精度，迭代停止。 最终代价：" + currentCostValue);
                    return newCoefficientList;
                }

                if (calculateTimes == maxLoop) {
                    System.out.println("达到最大迭代次数" + maxLoop + "，迭代停止。 最终代价" + currentCostValue);
                    return newCoefficientList;
                }
            }

            calculateTimes++;
            clearAndAddNewValue(coefficientList, newCoefficientList);
        }
    }

    /**
     * 计算代价函数结果
     * @param dataParams 数据集合
     * @param dataResults 结果集合
     * @param coefficientList 系数集合
     * @return 代价
     */
    private Double costFunctionResult(List<List<Double>> dataParams, List<Double> dataResults, List<Double> coefficientList) {

        double result = 0.0;

        for (int listIndex = 0; listIndex < dataParams.size(); listIndex++) {

            double hypothesisResult = calculateHypothesisResult(dataParams.get(listIndex), coefficientList);
            double deviationSquare = calculateDeviationSquare(hypothesisResult, dataResults.get(listIndex));

            result += deviationSquare;
        }

        return result/2/dataParams.size();
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
     * 计算新的结果系数list
     * @param coefficientList 之前的结果系数
     * @param dataParamsList 数据参数
     * @param dataResults 数据结果
     * @return 新的结果系数list
     */
    private List<Double> calculateNewCoefficientValueList(List<Double> coefficientList, List<List<Double>> dataParamsList, List<Double> dataResults) {

        List<Double> newCoefficientValueList = new ArrayList<>(coefficientList.size());

        for (int coefficientIndex = 0; coefficientIndex<coefficientList.size(); coefficientIndex++) {
            double newCoefficientValue = calculateNewCoefficientValue(coefficientList, coefficientIndex, dataParamsList, dataResults);
            newCoefficientValueList.add(newCoefficientValue);
        }

        return newCoefficientValueList;
    }

    /**
     * 计算新的结果系数值
     * @param coefficientList 之前的结果系数
     * @param coefficientIndex 系数在list中的索引
     * @param dataParamsList 数据参数
     * @param dataResults 数据结果
     * @return 新的结果系数值
     */
    private double calculateNewCoefficientValue(List<Double> coefficientList, int coefficientIndex, List<List<Double>> dataParamsList, List<Double> dataResults) {

        double changeAmount = 0.0;

        for (int listIndex = 0; listIndex < dataParamsList.size(); listIndex++) {
            changeAmount += calculateOneDifferentialTerms(dataParamsList.get(listIndex), dataResults.get(listIndex), coefficientList, coefficientIndex);
        }

        changeAmount = changeAmount*studyRate/dataParamsList.size();

        return coefficientList.get(coefficientIndex) - changeAmount;
    }

    /**
     * 计算一个差值项的平方
     * @param dataParams 一条数据
     * @param result 数据的实际结果
     * @param coefficientList 系数列表
     * @param coefficientIndex 系数在list中的索引
     * @return result
     */
    private double calculateOneDifferentialTerms(List<Double> dataParams, double result, List<Double> coefficientList, int coefficientIndex) {

        double hypothesisResult = calculateHypothesisResult(dataParams, coefficientList);

        return (hypothesisResult - result)*dataParams.get(coefficientIndex);
    }

    /**
     * 降低学习速率并恢复之前的结果系数
     * @param newCoefficientList 新的结果系数
     * @param coefficientList 之前的结果系数
     */
    private void declineStudyRateAndRecoverCoefficient(List<Double> newCoefficientList, List<Double> coefficientList) {
        this.studyRate *= declineValue;
        clearAndAddNewValue(newCoefficientList, coefficientList);
    }

    /**
     * 将另外一个list的值全部放入目标list中
     * @param destination 目标list
     * @param resource 值来源list
     * @param <T> 泛型
     */
    private <T> void clearAndAddNewValue(List<T> destination, List<T> resource) {

        destination.clear();
        destination.addAll(resource);
    }

    /**
     * 使用正规方程计算线性回归方程参数
     * @param dataParamsList 参数集合，按照相同顺序组成的列表的集合,size()>1
     * @param dataResults 结果集合,一个参数列表的索引和对应结果在结果集合中的索引相同
     */
    private Matrix calculateRegressionResultWithNormalEquation(List<List<Double>> dataParamsList, List<Double> dataResults) {

        Matrix paramsMatrix = getParamsMatrix(dataParamsList);
        Matrix resultsMatrix = createMatrixWithList(dataResults);

        Matrix transposeMatrix = paramsMatrix.transpose();

        Matrix multiplyMatrix = transposeMatrix.mtimes(paramsMatrix);

        Matrix inverseMatrix = multiplyMatrix.ginv();

        Matrix eye = inverseMatrix.mtimes(multiplyMatrix);

        multiplyMatrix = inverseMatrix.mtimes(transposeMatrix);

        return multiplyMatrix.mtimes(resultsMatrix);

    }

//    private RealMatrix calculateResult(List<List<Double>> dataParamsList, List<Double> dataResults) {
//
//        RealMatrix dataMatrix = getRealMatrix(dataParamsList);
//        RealMatrix resultMatrix = getResulRealtMatrix(dataResults);
//
//        RealMatrix transposeMatrix = dataMatrix.transpose();
//
//        RealMatrix multiplyMatrix = transposeMatrix.multiply(dataMatrix);
//
//        RealMatrix inverseMatrix = multiplyMatrix.
//
//        multiplyMatrix = inverseMatrix.multiply(transposeMatrix);
//
//        return multiplyMatrix.multiply(resultMatrix);
//    }

//    private RealMatrix getRealMatrix(List<List<Double>> dataParamsList) {
//
//        int listSize = dataParamsList.get(0).size();
//
//        double[][] dataParamArray = new double[dataParamsList.size()][listSize+1];
//
//        for (int listIndex = 0; listIndex < dataParamsList.size(); listIndex++) {
//            dataParamArray[listIndex][0] = 1.0;
//
//            for (int valueIndex = 0; valueIndex < listSize; valueIndex++) {
//                dataParamArray[listIndex][valueIndex+1] = dataParamsList.get(listIndex).get(valueIndex);
//            }
//        }
//
//        return new Array2DRowRealMatrix(dataParamArray);
//    }
//
//    private Matrix getMatrix(List<List<Double>> dataParamsList) {
//
//        int listSize = dataParamsList.get(0).size();
//
//        double[][] dataParamArray = new double[dataParamsList.size()][listSize+1];
//
//        for (int listIndex = 0; listIndex < dataParamsList.size(); listIndex++) {
//            dataParamArray[listIndex][0] = 1.0;
//
//            for (int valueIndex = 0; valueIndex < listSize; valueIndex++) {
//                dataParamArray[listIndex][valueIndex+1] = dataParamsList.get(listIndex).get(valueIndex);
//            }
//        }
//
//        return new Matrix(dataParamArray);
//    }

//    private RealMatrix getResulRealtMatrix(List<Double> dataResults) {
//
//        double[][] result = new double[dataResults.size()][1];
//
//        for (int value = 0; value < dataResults.size(); value++) {
//            result[value][0] = dataResults.get(value);
//        }
//
//        return new Array2DRowRealMatrix(result);
//    }
//
//    private Matrix getResultMatrix(List<Double> dataResults) {
//
//        double[][] result = new double[dataResults.size()][1];
//
//        for (int value = 0; value < dataResults.size(); value++) {
//            result[value][0] = dataResults.get(value);
//        }
//
//        return new Matrix(result);
//    }

//    private Matrix calculateInverse(Matrix matrix) {
//
//        SingularValueDecomposition svd = matrix.svd();
//        Matrix s = svd.getS();
//        Matrix v = svd.getV().transpose();
//        Matrix u = svd.getU();
//        //将S中非0元素取倒数
//        Matrix sinv = unaryNotZeroElement(s);
//        Matrix inv = v.times(sinv).times(u.transpose());
//
//        return inv;
//    }
//
//    /**
//     * 将矩阵中非0元素取倒数
//     * @param x 矩阵
//     * @return 计算结果
//     */
//    private Matrix unaryNotZeroElement(Matrix x) {
//        double[][] array=x.getArray();
//        for(int i=0;i<array.length;i++){
//            for(int j=0;j<array[i].length;j++){
//                if(array[i][j]!=0){
//                    array[i][j]=1.0/array[i][j];
//                }
//            }
//        }
//        return new Matrix(array);
//    }

    private void printRealMatrix(RealMatrix matrix) {

        long rows = matrix.getRowDimension();
        long columns = matrix.getColumnDimension();

        for (int row = 0; row < rows; row++) {
            for (int column = 0; column< columns; column++) {
                System.out.println(matrix.getEntry(row,column));
            }
        }
    }
}