package utils;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation;

/**
 * @author Ning
 * @date Create in 2018/6/22
 */
public class Functions {

    /**
     * sigmoid/logistic函数，公式y = 1/(1 + e^-x) y∈[0,1]，如果使用ujmp矩阵库计算，可以使用 matrix.logistic(Ret var1) 方法直接求矩阵的sigmoid结果
     * 用于二分类；缺点计算量大，反向传播时，容易出现梯度消失
     * @param value 要计算的值
     * @return 计算结果
     */
    public static double sigmoid(double value) {

        return 1 / (1 + Math.pow(Math.E, -value));
    }

    /**
     * 双曲正切函数，公式：(e^x-e^-x)/(e^x+e^-x) y∈[-1,1]
     * 特征明显时效果较好
     * @param value 要计算的值
     * @return 计算结果
     */
    public static double tanh(double value) {

        double ePowerNegativeX = Math.pow(Math.E, -value);
        double ePowerX = Math.pow(Math.E, value);

        return (ePowerX - ePowerNegativeX)/(ePowerX + ePowerNegativeX);
    }

    /**
     * ReLu，公式 max(0,x)
     * 训练速度快；缺点当学习速率过大或梯度过大时，会导致不更新权重值，需要将学习速率设置的低一些
     * @param value 要计算的值
     * @return 计算结果
     */
    public static double reLu(double value) {
        return Math.min(0.0, value);
    }

    /**
     * softMax, 公式σ(z)j = (e^z(j))/(∑(i=1，k)(e^z(i)))，矩阵的行数为k，即任意一项的结果为：e的该项值次幂除以e的全部值次幂之和
     * 用于多分类问题，输出每类可能出现的概率大小，和为1
     * @param coefficient 需要处理的矩阵，必须为向量矩阵
     * @return 处理后的结果
     */
    public static Matrix softMax(Matrix coefficient) {

        long column = coefficient.getColumnCount();

        if (column != 1) {
            System.out.println("softMax的参数不是向量矩阵，请检查");
            return null;
        }

        long rowCount = coefficient.getRowCount();

        Matrix baseMatrix  = constructValueEMatrix(rowCount);
        Matrix powerEMatrix = baseMatrix.power(Calculation.Ret.NEW, coefficient);

        return calculateResultMatrix(powerEMatrix);
    }

    /**
     * 正态分布概率 e^-((x-η)^2/2σ^2) / (√(2π)σ),这里用n替代η,用o替代σ
     * @param x x
     * @param n 期望
     * @param o 标准差
     * @return 概率
     */
    public static double normalDistribution(double x, double n, double o) {

        // (x-η)^2/2σ^2)
        double molecular = (x - n)*(x - n)/(2*o*o);

        // e^-((x-η)^2/2σ^2)
        molecular = StrictMath.pow(Math.E, -molecular);

        // 分母 √(2π)*σ
        double denominator = StrictMath.sqrt(2 * Math.PI)*o;

        return molecular/denominator;
    }

    /**
     * 构建指定行数的，值全部为e的向量
     * @param rowCount 行数
     * @return 构建结果矩阵
     */
    private static Matrix constructValueEMatrix(long rowCount) {

        Matrix baseMatrix  = Matrix.Factory.zeros(rowCount, 1);

        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            baseMatrix.setAsDouble(Math.E, rowCount, 1);
        }

        return baseMatrix;
    }

    /**
     * 计算softMax的结果矩阵
     * @param resultMatrix 以原始向量矩阵的值为底数,e次幂的向量矩阵
     * @return softMax的结果矩阵
     */
    private static Matrix calculateResultMatrix(Matrix resultMatrix) {

        long rowCount = resultMatrix.getRowCount();
        double molecular = resultMatrix.getValueSum();

        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {

            double originalValue = resultMatrix.getAsDouble(rowCount, 1);
            resultMatrix.setAsDouble(originalValue/molecular, rowCount, 1);
        }

        return resultMatrix;
    }
}
