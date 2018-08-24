package algorithm.gradient.descent;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
abstract class BaseNatureNetworkAlgorithm extends BaseDataConstructForNatureNetwork {

    /**
     * 激活函数
     * @param matrix 需要处理的矩阵
     * @return 处理结果
     */
    abstract Matrix activationFunction(Matrix matrix);


    /**
     * 迭代计算各层权重系数矩阵
     */
    void calculateNatureNetworkResult() {

        int cycleTimes = 0;

        while (true) {

            double previousCost = calculateCostForNatureNetWork();

            List<Matrix> predictionErrorMatrices = calculatePredictionErrorMatrices();

            List<Matrix> weightGradientMatrices = calculateWeightGradientMatrices(predictionErrorMatrices);

            List<Matrix> weightUpdateMatrices = calculateWeightUpdateMatrices(weightGradientMatrices);

            updateCoefficientMatrices(weightUpdateMatrices);

            double newCost = calculateCostForNatureNetWork();

            if (newCost - previousCost < convergence) {

                System.out.println("达到目标精确值，迭代停止，迭代次数：" + cycleTimes + "，最终去除正则部分代价：" + calculateFinalCostWithoutRegulationPart(newCost));
                Utils.printList(hideCoefficientMatrices);
                break;

            } else if (cycleTimes > maxLoop) {
                System.out.println("达到最大迭代次数，迭代停止，最终去除正则部分代价：" + calculateFinalCostWithoutRegulationPart(newCost));
                Utils.printList(hideCoefficientMatrices);
                break;
            }

            cycleTimes++;
        }
    }


    // =========================神经网络代价相关计算，包括记录每层添加偏置项的结果=========================

    /**
     * 计算神经网络算法的代价值
     * @return 代价值
     */
    private double calculateCostForNatureNetWork() {

        // 清理之前计算的每层假设结果，以便添加新的假设结果
        hypothesisMatrices.clear();
        double hypothesisCostValue = calculateHypothesisPartValue();

        double regulationCostValue = calculateRegulationPartValue();

        return -(hypothesisCostValue - regulationCostValue)/dataMatrix.getRowCount();

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
     * 计算从输入层到最后一层的结果，顺便保存每层的假设结果，供反向传播使用
     * @return 结果
     */
    private Matrix matrixMultiply() {

        Matrix resultMatrix = dataMatrix;

        for (int index = 0; index < hideCoefficientMatrices.size(); index++) {

            // 激活后的结果
            resultMatrix = calculateActivatingResult(resultMatrix, index);
        }

        hypothesisMatrices.add(resultMatrix);
        return resultMatrix;
    }

    /**
     * 计算当前层激活函数处理后的结果矩阵
     * @param lastPostActivatingResult 当前层激活后的结果，第一个值为输入值
     * @param hideLayerNumber 当前层数，0-输出层前一层，数量=隐藏层个数
     * @return 激活函数处理后的下一层结果
     */
    private Matrix calculateActivatingResult(Matrix lastPostActivatingResult, int hideLayerNumber) {

        // 第一层的数据，即原始数据已经添加过偏置项，无需再次添加
        if (hideLayerNumber != 0) {
            lastPostActivatingResult = addOneToList(lastPostActivatingResult);
        }
        hypothesisMatrices.add(lastPostActivatingResult);

        Matrix currentLayerActivatingMatrix = hideCoefficientMatrices.get(hideLayerNumber);

        return activationFunction(lastPostActivatingResult.times(currentLayerActivatingMatrix));
    }

    /**
     * 给激活值矩阵值前添加一个值均为1的常数向量，使其能和带有常数项的系数矩阵相乘
     * @param matrix 激活矩阵
     */
    private Matrix addOneToList(Matrix matrix) {

        Matrix oneMatrix = Matrix.Factory.zeros(matrix.getRowCount(), 1);
        oneMatrix.setAsDouble(1.0, 0, 0);
        return oneMatrix.appendVertically(Calculation.Ret.NEW, matrix);
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

    /**
     * 计算最终不包含正则部分的代价
     * @param finalCost 正则代价
     * @return 去除正则部分的代价
     */
    private double calculateFinalCostWithoutRegulationPart(double finalCost) {

        double finalRegulationCostValue = calculateRegulationPartValue();

        return finalCost - finalRegulationCostValue/dataMatrix.getRowCount();
    }



    // =========================神经网络反向传播相关计算=========================

    /**
     * 计算每层的预测误差，由于反向传播的特点，第一个预测误差为输出层的预测误差，最后一个预测误差为第一层隐藏层的预测误差，共总层数-1个
     * @return 每层的预测误差δ
     */
    private List<Matrix> calculatePredictionErrorMatrices() {

        List<Matrix> predictionErrorMatrices = new ArrayList<>();

        Matrix lastLayerPredictionError = hypothesisMatrices.get(hideCoefficientMatrices.size()-1).minus(resultMatrix);

        predictionErrorMatrices.add(lastLayerPredictionError);

        for (int index = hideCoefficientMatrices.size() - 1; index > 0; index--) {

            // 获取最近一个，即上一次算出来的预测误差矩阵
            Matrix latestPredictionError = predictionErrorMatrices.get(predictionErrorMatrices.size() - 1);

            Matrix currentLayerCoefficientMatrix = hideCoefficientMatrices.get(index);

            // 获取当前层误差差值，和系数矩阵位于同一层
            Matrix currentLayerHypothesisMatrix = hypothesisMatrices.get(index);

            Matrix currentLayerHypothesisUnitMatrix = Matrix.Factory.zeros(currentLayerHypothesisMatrix.getRowCount(), currentLayerHypothesisMatrix.getColumnCount());

            // 激活函数的导数 g'(z(l)) = a(l) .* (1 - a(l))
            Matrix activeFunctionDerivative = currentLayerHypothesisMatrix.times(currentLayerHypothesisUnitMatrix.minus(currentLayerHypothesisMatrix));

            // δ(l) = δ(l+1) * θ(l)^T .* g'(z(l))
            Matrix predictionErrorMatrix = latestPredictionError.mtimes(currentLayerCoefficientMatrix.transpose()).times(activeFunctionDerivative);

            predictionErrorMatrices.add(predictionErrorMatrix);
        }

        return predictionErrorMatrices;
    }

    /**
     * 计算从输出层前一层到输入层的各层权值梯度
     * @param predictionErrorMatrices 输出层到输入层各层预测误差，逆序
     * @return 各层权值梯度，顺序
     */
    private List<Matrix> calculateWeightGradientMatrices(List<Matrix> predictionErrorMatrices) {

        List<Matrix> weightGradientMatrices = new ArrayList<>();

        for (int index = predictionErrorMatrices.size() - 1; index > -1; index--) {

            // a的索引 = a.size() - 比预测误差size多的1 - 预测误差的索引 - 索引修正值
            int hypothesisMatrixIndex = hypothesisMatrices.size() - 1 - index - 1;

            // Δ(l) = (a(l))^T * δ(l + 1)
            weightGradientMatrices.add(hypothesisMatrices.get(hypothesisMatrixIndex).transpose().mtimes(predictionErrorMatrices.get(index)));
        }

        return weightGradientMatrices;
    }

    /**
     * 计算每层系数矩阵的更新增量D
     * @param weightGradientMatrices 各层权值梯度，顺序
     * @return 权值更新量，顺序
     */
    private List<Matrix> calculateWeightUpdateMatrices(List<Matrix> weightGradientMatrices) {

        List<Matrix> weightUpdateMatrices = new ArrayList<>();

        for (int index =  0; index < weightGradientMatrices.size(); index++) {

            Matrix ignoreRegulationItemCoefficientMatrix = getIgnoreRegulationItemCoefficientMatrix(index);

            // D(l) = ( Δ(l) + λθ(l) )/m
            Matrix weightUpdateMatrix = (weightGradientMatrices.get(index).plus(ignoreRegulationItemCoefficientMatrix.times(lambda))).divide(dataMatrix.getRowCount());

            weightUpdateMatrices.add(weightUpdateMatrix);
        }

        return weightUpdateMatrices;
    }

    /**
     * 获取指定索引的忽略正则项，即将偏置项设置为0的的系数矩阵
     * @param coefficientIndex 系数矩阵索引
     * @return 不带偏置项的系数矩阵
     */
    private Matrix getIgnoreRegulationItemCoefficientMatrix(int coefficientIndex) {

        Matrix resultMatrix = hideCoefficientMatrices.get(coefficientIndex).clone();

        for (int columnIndex = 0; columnIndex < resultMatrix.getColumnCount(); columnIndex++) {
            resultMatrix.setAsDouble(0.0, 0, columnIndex);
        }
        return resultMatrix;
    }

    /**
     * 使用权值更新量更新每层的系数矩阵
     * @param weightUpdateMatrices 各层权值的更新量，顺序
     */
    private void updateCoefficientMatrices(List<Matrix> weightUpdateMatrices) {

        for (int index = 0; index < weightUpdateMatrices.size(); index++) {

            Matrix weightUpdateMatrix = weightUpdateMatrices.get(index);
            Matrix originalCoefficientMatrix = hideCoefficientMatrices.get(index);

            originalCoefficientMatrix.plus(weightUpdateMatrix.times(studyRate));
        }
    }
}
