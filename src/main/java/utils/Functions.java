package utils;

/**
 * @author Ning
 * @date Create in 2018/6/22
 */
public class Functions {

    /**
     * sigmoid函数，公式y = 1/(1 + e^-x)
     * @param value 要计算的值
     * @return 计算结果
     */
    public static double sigmoid(double value) {

        return 1 / (1 + Math.pow(Math.E, -value));
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
}
