package algorithm.gradient.descent;

import org.ujmp.core.Matrix;

import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
abstract class BaseNatureNetworkAlgorithm extends BaseDataConstructForNatureNetwork {

    private List<Integer> hideUnitsAmount;

    BaseNatureNetworkAlgorithm(List<Double> dataList, List<Integer> hideUnitsAmount) {
        super(dataList, hideUnitsAmount);
    }

    /**
     * 激活函数
     * @param matrix 需要处理的矩阵
     * @return 处理结果
     */
    abstract Matrix activationFunction(Matrix matrix);


    double calculateNatureNetworkResult(List<Double> dataParams) {

        dataMatrix = createMatrixWithList(dataParams);

        return matrixMultiply(hideCoefficientMatrices.size()-1).getAsDouble(0, 0);
    }

    /**
     * 递归计算从输入值到最后一层的输出矩阵
     * @param index 最后一层的索引
     * @return 结果
     */
    private Matrix matrixMultiply(int index) {

        if (index == 0) {
            return activationFunction(hideCoefficientMatrices.get(index).times(dataMatrix));

        } else {
            return activationFunction(hideCoefficientMatrices.get(index).times(matrixMultiply(--index)));
        }
    }

    /**
     * 计算下一层激活值矩阵
     * @param lastActivatingResult 上一层激活值结果
     * @param hideLayerNumber 当前层数
     * @return 激活结果
     */
    private Matrix calculateActivatingResult(Matrix lastActivatingResult, int hideLayerNumber) {

        Matrix currentLayerActivatingMatrix = hideCoefficientMatrices.get(hideLayerNumber);

        return activationFunction(currentLayerActivatingMatrix.times(lastActivatingResult));
    }
}
