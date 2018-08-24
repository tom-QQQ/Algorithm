package algorithm.gradient.descent;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;
import utils.Functions;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
public class NatureNetwork extends BaseNatureNetworkAlgorithm {

    /**
     * sigmoid:1, ReLu:2, tanH:3
     */
    private int activationFunctionType;

    private static final String SIGMOID = "sigmoid";
    private static final String RE_LU = "ReLu";
    private static final String TAN_H = "tanH";

    {
        // 神经网络设置是否需要规格化数据
        ifNeedNormalization = false;
    }


    public NatureNetwork(String activationFunctionName) {

        if (SIGMOID.equals(activationFunctionName) ) {
            this.activationFunctionType = 1;

        } else if (RE_LU.equals(activationFunctionName)) {
            this.activationFunctionType = 2;

        } else if (TAN_H.equals(activationFunctionName)) {
            this.activationFunctionType = 3;

        } else {

            System.out.println("激活函数名输入有误！！！");
            System.exit(0);
        }

    }

    /**
     * 计算测试数据的结果
     */
    public void calculateExampleResult() {

        List<List<Double>> dataList = new ArrayList<>();
        addDataToList(dataList, 0.0, 0.0);
        addDataToList(dataList, 0.0, 1.0);
        addDataToList(dataList, 1.0, 1.0);
        addDataToList(dataList, 1.0, 1.0);


        List<List<Double>> resultLists = new ArrayList<>();
        addDataToList(dataList, 1.0);
        addDataToList(dataList, 0.0);
        addDataToList(dataList, 0.0);
        addDataToList(dataList, 1.0);

        List<Integer> hideUnitsNeuronsAmounts = new ArrayList<>();
        hideUnitsNeuronsAmounts.add(2);
        hideUnitsNeuronsAmounts.add(1);

        try {

            dataConstructForNatureNetwork(dataList, hideUnitsNeuronsAmounts, resultLists);

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }

        calculateNatureNetworkResult();

    }

    @Override
    Matrix activationFunction(Matrix matrix) {

        if (activationFunctionType == 1) {
            return matrix.logistic(Calculation.Ret.NEW);

        } else {
            return changeBySpecificFunction(matrix);
        }
    }

    /**
     * 根据激活函数类型调用不同的激活函数
     * @param matrix 原始矩阵
     * @return 结果矩阵
     */
    private Matrix changeBySpecificFunction(Matrix matrix) {

        long rowCount = matrix.getRowCount();
        long columnCount = matrix.getColumnCount();
        Matrix resultMatrix = Matrix.Factory.zeros(rowCount, columnCount);

        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {

            for (int columnIndex = 0; columnIndex < columnCount; columnIndex++) {

                double originalValue = matrix.getAsDouble(rowIndex,columnIndex);
                double newValue;

                if (activationFunctionType == 2) {
                    newValue = Functions.reLu(originalValue);

                } else {
                    newValue = Functions.tanH(originalValue);
                }
                resultMatrix.setAsDouble(newValue, rowIndex, columnIndex);
            }
        }
        return resultMatrix;
    }
}
