package algorithm.gradient.descent;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/24
 */
class BaseExamine {

    /**
     * 是否需要规格化数据
     */
    static boolean ifNeedNormalization;

    /**
     * 神经网络的多个隐藏层对应的系数矩阵,包括最后一层隐藏层到结果层的系数矩阵，该层矩阵为m*1(最后一层隐藏层神经单元个数)
     */
    List<Matrix> hideCoefficientMatrices;

    List<Double> averageValues = new ArrayList<>();
    List<Double> squaredDifferenceValues = new ArrayList<>();

    private Matrix dataMatrix;


    /**
     * 规格化检验数据，应在调用回归方法计算数据的平均值和方差后调用，需要在数据之前添加一个1.0
     * @param dataParams 数据
     */
    void normalListValue(List<Double> dataParams) {

        if (ifNeedNormalization) {
            for (int valueIndex = 1; valueIndex < dataParams.size(); valueIndex++) {
                double value = dataParams.get(valueIndex);
                dataParams.set(valueIndex, (value - this.averageValues.get(valueIndex - 1))/this.squaredDifferenceValues.get(valueIndex - 1));
            }
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
