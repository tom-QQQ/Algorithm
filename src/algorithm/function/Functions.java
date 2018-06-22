package algorithm.function;

/**
 * @author Ning
 * @date Create in 2018/6/22
 */
public class Functions {

    /**
     * sigmoid函数，公式y = 1/(1-e^-x)
     * @param value 要计算的值
     * @return 计算结果
     */
    public static double sigmoid(double value) {

        return 1 / (1 - Math.pow(Math.E, -value));
    }
}
