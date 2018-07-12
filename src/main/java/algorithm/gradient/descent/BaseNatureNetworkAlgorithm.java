package algorithm.gradient.descent;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

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

        return matrixMultiply();
    }

    /**
     * 计算从输入值到最后一层的结果
     * @return 结果
     */
    private double matrixMultiply() {

        // 输入的数据矩阵已经做第一项前加1处理，这里系数矩阵可以直接和其相乘
        Matrix resultMatrix = hideCoefficientMatrices.get(0).mtimes(dataMatrix);

        for (int index = 1; index < hideCoefficientMatrices.size(); index++) {

            resultMatrix = addOneToList(resultMatrix);
            resultMatrix = hideCoefficientMatrices.get(index).times(resultMatrix);
        }

        return resultMatrix.getAsDouble(0, 0);
    }

    /**
     * 计算下一层激活值矩阵
     * @param lastActivatingResult 上一层激活值结果
     * @param hideLayerNumber 当前层数
     * @return 激活结果
     */
    private Matrix calculateActivatingResult(Matrix lastActivatingResult, int hideLayerNumber) {

        lastActivatingResult = addOneToList(lastActivatingResult);

        Matrix currentLayerActivatingMatrix = hideCoefficientMatrices.get(hideLayerNumber);

        return activationFunction(currentLayerActivatingMatrix.times(lastActivatingResult));
    }

    /**
     * 给激活矩阵的值前添加一个常数项1
     * @param matrix 激活矩阵
     */
    private Matrix addOneToList(Matrix matrix) {

        Matrix oneMatrix = Matrix.Factory.zeros(1, 1);
        oneMatrix.setAsDouble(1.0, 0, 0);
        return oneMatrix.appendHorizontally(Calculation.Ret.NEW, matrix);
    }
}
