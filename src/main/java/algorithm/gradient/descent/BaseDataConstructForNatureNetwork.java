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
     * 该矩阵包含了全部数据的值，m*(1+n)，m为数据条数，1为常数项1.0，n为每条数据的参数数量
     */
    Matrix dataMatrix;

    BaseDataConstructForNatureNetwork(List<List<Double>> dataList, List<Integer> hideUnitsNeuronsAmounts) {


        // 这里的-1是为了和之后的保持一致，下面的方法返回的结果行会比之前加1(常数项)
        int row = dataList.get(0).size() - 1;

        for (int unitIndex = 0; unitIndex < hideUnitsNeuronsAmounts.size() - 1; unitIndex++) {

            int column = hideUnitsNeuronsAmounts.get(unitIndex);
            Matrix hideUnitCoefficient = constructMatrixWithRowPlusOne(row, column);
            hideCoefficientMatrices.add(hideUnitCoefficient);
            row = column;
        }
    }

    private void noramlizationData(List<List<Double>> dataList) {

        List<List<Double>> normalizationList;

        // 此时dataList的size比之前+1
        if (ifNeedNormalization) {
            normalizationList = calculateNormalizationData(dataList);

        } else {
            normalizationList = dataList;
        }


        this.dataMatrix = constructDataMatrix(normalizationList);

    }

    /**
     * 创建一个指定行+1(当前层的常数项)，指定列的系数矩阵，值为指定范围中的值
     * @param rowCount 下一层单元个数
     * @param columnCount 当前层单元个数
     * @return 当前层到下一层的计算系数矩阵
     */
    private Matrix constructMatrixWithRowPlusOne(long rowCount, long columnCount) {

        Matrix coefficientMatrix = Matrix.Factory.zeros(rowCount + 1, columnCount);
        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {

            for (int columnIndex = 0; columnIndex < columnCount; columnIndex++) {

                coefficientMatrix.setAsDouble(getNumberInSpecificRange());
            }
        }

        return coefficientMatrix;
    }



    //========================以下为规格化List<Double>类型数据的规格方法=============================


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
