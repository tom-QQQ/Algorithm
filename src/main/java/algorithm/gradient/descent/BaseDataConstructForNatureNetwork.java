package algorithm.gradient.descent;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
class BaseDataConstructForNatureNetwork extends BaseDataConstruct {

    /**
     * 该矩阵包含了全部数据的值，m*(1+n)，m为数据条数，1对应常数项，n为每条数据的参数数量
     */
    Matrix dataMatrix;

    /**
     * 数据结果矩阵，该矩阵的行列数和神经网络最后得到的结果矩阵的行列数必须相同，才可以进行对应值的计算
     */
    Matrix resultMatrix;

    /**
     * 反向传播时需要用到包含偏置项的每层的值和最后一层的不包含偏置项的结果层的值,size = 总层数(包括输入层和输出层)个数
     */
    List<Matrix> hypothesisMatrices = new ArrayList<>();

    /**
     * 数据构建方法，用于子类调用初始化数据
     * @param dataLists 输入数据
     * @param hideUnitsNeuronsAmounts 隐藏层各层的神经元数量
     * @param resultLists 目标结果
     * @throws Exception
     */
    void dataConstructForNatureNetwork(List<List<Double>> dataLists, List<Integer> hideUnitsNeuronsAmounts, List<List<Double>> resultLists) throws
            Exception {

        if (dataLists.size() != resultLists.size() || hideUnitsNeuronsAmounts.get(hideUnitsNeuronsAmounts.size() - 1) != resultLists.get(0).size()) {
            throw new Exception("参数数量和结果数量不同，或最后一层神经元数量和结果种类数量不同");
        }

        normalizationAndSaveData(dataLists);

        // 这里的-1是为了和之后的保持一致，下面的循环方法返回的结果列会比之前加1(增加偏置项)
        int row = dataLists.get(0).size() - 1;

        for (int column : hideUnitsNeuronsAmounts) {

            Matrix hideUnitCoefficient = constructMatrixWithRowPlusOne(row, column);
            hideCoefficientMatrices.add(hideUnitCoefficient);
            row = column;
        }

        this.resultMatrix = constructDataMatrix(resultLists);
    }

    /**
     * 正则化并保存数据，会给每个list第一项前加一项偏置值
     * @param dataList 输入的数据
     */
    private void normalizationAndSaveData(List<List<Double>> dataList) {

        List<List<Double>> normalizationList;

        // 此时dataList的size比之前+1
        if (ifNeedNormalization) {
            normalizationList = calculateNormalizationData(dataList);

        } else {
            normalizationList = addOneToLists(dataList);
        }

        this.dataMatrix = constructDataMatrix(normalizationList);

    }

    /**
     * 创建一个指定行+1(当前层的常数项，第一项)，指定列的系数矩阵，值为指定范围中的值
     * @param rowCount 下一层单元个数
     * @param columnCount 当前层单元个数
     * @return 当前层到下一层的计算系数矩阵
     */
    private Matrix constructMatrixWithRowPlusOne(long rowCount, long columnCount) {

        Matrix coefficientMatrix = Matrix.Factory.zeros(rowCount + 1, columnCount);
        for (int rowIndex = 0; rowIndex < rowCount + 1; rowIndex++) {

            for (int columnIndex = 0; columnIndex < columnCount; columnIndex++) {

                coefficientMatrix.setAsDouble(getNumberInSpecificRange());
            }
        }

        return coefficientMatrix;
    }

    /**
     * 给不需要规格化的每项数据list增加一个偏置项1.0
     * @param dataList 未规格化的原始数据
     * @return 每项前添加偏置值的新lists
     */
    private List<List<Double>> addOneToLists(List<List<Double>> dataList) {

        List<List<Double>> resultList = new ArrayList<>();

        for (List<Double> list : dataList) {

            List<Double> originalList = new ArrayList<>();
            originalList.add(1.0);
            originalList.addAll(list);
            resultList.add(originalList);
        }

        return resultList;
    }



    //=============================以下为规格化List<Double>类型数据的规格方法=============================


    /**
     * 规格化数据并构建矩阵,行数为之前行数+1，在第一个数值之前添加添加常数项1
     * @param dataList 原始数据
     * @return 规格化矩阵，第一个值为常熟项1.0
     */
    private Matrix getNormalizationDataMatrix(List<Double> dataList) {

        List<Double> normalizationDataList = calculateSingleListNormalizationData(dataList);

        return createMatrixWithList(normalizationDataList);
    }

    /**
     * 规格化单list数据，并在规格数据前添加常数项1.0
     * @param dataList 需要规格化的数据
     * @return 规格化结果
     */
    private List<Double> calculateSingleListNormalizationData(List<Double> dataList) {

        double average = listAverageValue(dataList);
        double standardDeviation = calculateStandardDeviation(dataList, average);

        return calculateNormalizationResult(average, standardDeviation, dataList);

    }

    /**
     * 求list中全部值的平均数
     * @param dataList 需要求平均数的list
     * @return 平均数
     */
    private double listAverageValue(List<Double> dataList) {

        double result = 0.0;

        for (Double value : dataList) {
            result += value;
        }
        return result/dataList.size();
    }

    /**
     * 计算list的方差
     * @param dataList 数据
     * @param summation 平均值
     * @return 方差
     */
    private double calculateStandardDeviation(List<Double> dataList, double summation) {

        double standardDeviation = 0.0;

        for (Double value : dataList) {

            standardDeviation += (value - summation)*(value - summation);
        }
        return Math.sqrt(standardDeviation/dataList.size());
    }

    /**
     * 计算规格化结果,并设置第一项的值为1
     * @param average 平局值
     * @param standardDeviation 标准差
     * @param dataList 需要规格化的数据
     * @return 规格化结果
     */
    private List<Double> calculateNormalizationResult(double average, double standardDeviation, List<Double> dataList) {

        List<Double> normalizationResult = new ArrayList<>();
        normalizationResult.add(1.0);

        for (Double originalValue : dataList) {
            normalizationResult.add((originalValue - average)/standardDeviation);
        }

        return normalizationResult;
    }
}
