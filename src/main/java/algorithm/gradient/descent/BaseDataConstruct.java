package algorithm.gradient.descent;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/24
 */
class BaseDataConstruct extends BaseExamine {

    int initialNumberRange;
    double studyRate;
    double declineValue;
    long maxLoop;
    double convergence;

    /**
     * 正则系数，为0时不进行正则化,当正则系数较小但仍然大于0时，需要迭代计算更多次，但更容易发现相关性较小的参数，但再向0接近时，迭代次数又会减少
     * 如果正则系数较大，则代价也会增加很多，推荐范围[0.005, 1]，该值较大时，即使去除代价的正则部分，代价依然很大，此时需要更多的参数，如平方项等
     */
    double lambda;

    {
        initialNumberRange = 40;
        studyRate = 1.0;
        declineValue = 0.618;
        maxLoop = 1000000000L;
        convergence = 0.0001;
        lambda = 0.0;
        ifNeedNormalization = true;
    }

    /**
     * 生成指定范围内的随机值
     * @return 指定范围的随机值
     */
    double getNumberInSpecificRange() {

        double number = Math.random()*initialNumberRange;

        boolean ifNegative = Math.random() < 0.5;

        return (ifNegative ? -number : number);
    }

    /**
     * 根据结果数据创建对应矩阵，该方法需要在子类中调用
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
     * 根据完成规格化的数据构造矩阵，该方法无需在子类中调用
     * @param dataParamsList 规格化完毕的数据
     * @return 对应矩阵 m*n
     */
        Matrix constructDataMatrix(List<List<Double>> dataParamsList) {

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
     * 随机初始化系数结果的值，该方法无需在子类中调用
     * @param coefficientsSize 系数个数
     * @return 初始结果系数List
     */
    List<Double> initCoefficientList(int coefficientsSize) {

        List<Double> coefficientList = new ArrayList<>(coefficientsSize);

        for (int index = 0; index < coefficientsSize; index++) {
            coefficientList.add(getNumberInSpecificRange());
        }

        return coefficientList;
    }

    /**
     * 对原始数据进行规格化操作，并在每列数据前添加一个1，该方法无需在子类中调用
     * @param dataParamsList 原始数据
     * @return 规格化结果
     */
    List<List<Double>> calculateNormalizationData(List<List<Double>> dataParamsList) {

        List<Double> averageValues = new ArrayList<>();
        List<Double> squaredDifferenceList = new ArrayList<>();

        if (ifNeedNormalization) {

            averageValues = calculateAverageValueList(dataParamsList);
            this.averageValues.addAll(averageValues);


            squaredDifferenceList = calculateStandardDeviationList(dataParamsList, averageValues);
            this.squaredDifferenceValues.addAll(squaredDifferenceList);

            if (averageValues.size() != squaredDifferenceList.size() || averageValues.size() != dataParamsList.get(0).size()) {
                System.out.println("参数平均值数量和方差结果数量和参数list数量三者不相同");
                return null;
            }
        }

        return getNormalizationParamsList(dataParamsList, averageValues, squaredDifferenceList);
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
     * 计算给定List<>中List的标准差
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
     * @return 规格化结果,每个list在第一项之前增加一个1.0
     */
    private List<List<Double>> getNormalizationParamsList(List<List<Double>> dataValues, List<Double> averageValues, List<Double> squaredDifferenceList) {

        List<List<Double>> normalizationResults = initNormalizationResultsList(dataValues.size(), dataValues.get(0).size());

        for (int valueIndex = 0; valueIndex < dataValues.get(0).size(); valueIndex++) {

            for (int listIndex = 0; listIndex < dataValues.size(); listIndex++) {

                Double value = dataValues.get(listIndex).get(valueIndex);

                if (ifNeedNormalization) {
                    value = calculateNormalizationValue(value, averageValues.get(valueIndex), squaredDifferenceList.get(valueIndex));
                }

                normalizationResults.get(listIndex).add(value);
            }
        }

        return normalizationResults;
    }

    /**
     * 计算规格化值，公式(a-average)/方差
     * @param value 初始值
     * @param average 平均值
     * @param squaredDifference 方差
     * @return 规格化结果
     */
    private Double calculateNormalizationValue(Double value, Double average, Double squaredDifference) {

        return (value - average)/squaredDifference;
    }
}
