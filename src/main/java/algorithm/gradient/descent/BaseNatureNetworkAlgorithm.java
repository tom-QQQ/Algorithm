package algorithm.gradient.descent;

import com.sun.xml.internal.bind.v2.TODO;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
abstract class BaseNatureNetworkAlgorithm extends BaseDataConstructForNatureNetwork {

    private List<Integer> hideUnitsAmount;

    BaseNatureNetworkAlgorithm(List<List<Double>> dataList, List<Integer> hideUnitsAmount) {
        super(dataList, hideUnitsAmount);
    }

    /**
     * 激活函数
     * @param matrix 需要处理的矩阵
     * @return 处理结果
     */
    abstract Matrix activationFunction(Matrix matrix);


    Matrix calculateNatureNetworkResult() {


        return null;
    }

    /**
     * 计算从输入值到最后一层的结果
     * @return 结果
     */
    private Matrix matrixMultiply() {

        // 输入的数据矩阵已经做第一项前加1处理，这里单独计算
        Matrix resultMatrix = hideCoefficientMatrices.get(0).mtimes(dataMatrix);

        for (int index = 1; index < hideCoefficientMatrices.size(); index++) {

            resultMatrix = calculateActivatingResult(resultMatrix, index);
        }

        // 这里的计算结果应该是一个m*1的矩阵，
        return resultMatrix;
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

        return activationFunction(lastActivatingResult.times(currentLayerActivatingMatrix));
    }

    /**
     * 给激活矩阵的值前添加一个值均为1的常数向量，使其能和带有常数项的系数矩阵相乘
     * @param matrix 激活矩阵
     */
    private Matrix addOneToList(Matrix matrix) {

        Matrix oneMatrix = Matrix.Factory.zeros(matrix.getRowCount(), 1);
        oneMatrix.setAsDouble(1.0, 0, 0);
        return oneMatrix.appendVertically(Calculation.Ret.NEW, matrix);
    }
}
