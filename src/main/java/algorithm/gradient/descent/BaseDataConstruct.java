package algorithm.gradient.descent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/24
 */
class BaseDataConstruct extends BaseExamine {

    int initialNumberRange;

    /**
     * 给参数数组添加数据
     * @param dataParamsList 参数数组
     * @param values 一条数据
     */
    void addDataToDataParamsList(List<List<Double>> dataParamsList, Double...  values) {

        List<Double> list = Arrays.asList(values);
        dataParamsList.add(list);
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

    /**
     * 随机初始化系数结果的值，该方法无需在子类中调用
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
     * 生成指定绝对值范围内的随机值
     * @return 指定范围的随机值
     */
    private double createInitNumberInRange() {

        double number = Math.random()*initialNumberRange;

        boolean ifNegative = Math.random() < 0.5;

        return (ifNegative ? -number : number);
    }

    /**
     * 对原始数据进行规格化操作，并在每列数据添加一个1，该方法无需在子类中调用
     * @param dataParamsList 原始数据
     * @return 规格化结果
     */
    List<List<Double>> calculateNormalizationData(List<List<Double>> dataParamsList) {

        List<Double> averageValues = calculateAverageValueList(dataParamsList);
        this.averageValues.addAll(averageValues);

        List<Double> squaredDifferenceList = calculateStandardDeviationList(dataParamsList, averageValues);
        this.squaredDifferenceValues.addAll(squaredDifferenceList);

        if (averageValues.size() != squaredDifferenceList.size() || averageValues.size() != dataParamsList.get(0).size()) {
            System.out.println("参数平均值数量和方差结果数量和参数list数量三者不相同");
            return null;
        }

        return getNormalizationParamsList(dataParamsList, averageValues, squaredDifferenceList);
    }
}
