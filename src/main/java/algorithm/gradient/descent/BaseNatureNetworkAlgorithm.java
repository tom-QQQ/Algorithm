package algorithm.gradient.descent;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
abstract class BaseNatureNetworkAlgorithm extends BaseDataConstructForNatureNetwork {


    BaseNatureNetworkAlgorithm(List<List<Double>> dataList, List<Integer> hideUnitsAmount, List<List<Double>> resultLists) throws Exception {
        super(dataList, hideUnitsAmount, resultLists);
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
     * 计算神经网络算法的代价值
     * @return 代价值
     */
    private double calculateCostForNatureNetWork() {

        double hypothesisCostValue = calculateHypothesisPartValue();

        double regulationCostValue = calculateRegulationPartValue();

        return -(hypothesisCostValue - regulationCostValue)/dataMatrix.getRowCount();

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

        // 这里的计算结果应该是一个m*o(输出层神经元个数)的矩阵
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

    /**
     * 计算假设结果部分的值
     * @return 假设结果部分的值
     */
    private double calculateHypothesisPartValue() {

        Matrix hypothesisMatrix = matrixMultiply();

        for (int rowIndex = 0; rowIndex < resultMatrix.getRowCount(); rowIndex++) {

            for (int valueIndex = 0; valueIndex < resultMatrix.getColumnCount(); valueIndex++) {

                if (resultMatrix.getAsDouble(rowIndex, valueIndex) == 0) {

                    double newValue = 1 - hypothesisMatrix.getAsDouble(rowIndex, valueIndex);
                    hypothesisMatrix.setAsDouble(newValue, rowIndex, valueIndex);
                }
            }
        }

        hypothesisMatrix.log(Calculation.Ret.ORIG);

        return hypothesisMatrix.getValueSum();
    }

    /**
     * 计算神经网络算法中代价的正则部分,即不含偏置项的全部系数矩阵的平方和
     * @return 代价的正则部分
     */
    private double calculateRegulationPartValue() {

        double result = 0.0;

        for (Matrix matrix : hideCoefficientMatrices) {

            result += matrix.power(Calculation.Ret.NEW, 2.0).getValueSum();

            for (int columnIndex = 0; columnIndex < matrix.getColumnCount(); columnIndex++) {

                result -= Math.pow(matrix.getAsDouble(0, columnIndex), 2.0);
            }
        }

        return result;
    }
}
