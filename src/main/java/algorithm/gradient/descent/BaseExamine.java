package algorithm.gradient.descent;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/24
 */
class BaseExamine {

    List<Double> averageValues = new ArrayList<>();
    List<Double> squaredDifferenceValues = new ArrayList<>();


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
     * 计算假设结果
     * @param dataParams 一条数据
     * @param coefficientList 结果系数
     * @return 假设结果
     */
    double calculateHypothesisResult(List<Double> dataParams, List<Double> coefficientList) {

        double result = 0.0;
        for (int valueIndex = 0; valueIndex < dataParams.size(); valueIndex++) {
            result += dataParams.get(valueIndex)*coefficientList.get(valueIndex);
        }
        return result;
    }
}
