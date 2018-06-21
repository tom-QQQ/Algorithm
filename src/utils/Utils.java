package utils;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.TreeMap;

/**
 * @author Ning
 * @date Create in 2018/4/18
 */
public class Utils {

    /**
     * @param x x坐标
     * @param y y坐标
     * @return 规格化欧式距离
     */
    public static Double normalizationEuclideanDistance(List<Double> x, List<Double> y, boolean ifNeedNormalization) {

        if (x.size() != y.size()) {
            System.out.println("输入的两个数据长度不等，不能计算规格化欧式距离");
            return Double.NaN;
        }

        List<BigDecimal> normalizationX = calculateNormalization(x, ifNeedNormalization);
        List<BigDecimal> normalizationY = calculateNormalization(y, ifNeedNormalization);

        return  calculateEuclideanDistance(normalizationX, normalizationY);
    }

    /**
     * 计算规格化数据,不修改原值，这里只将结果处理∈[0，1]，计算公式: (ai - min)/(max - min)
     * @param list 需要规格化的数据
     * @param ifNeedNormalization 是否需要规格化，不需要就将Double转换为BigDecimal
     * @return 规格化结果,保留10位小数，结果位于[0,1]
     * 稍微修正一下规格化公式，由于最大值结果为1，最小值为0，可能造成原值不同而规格化结果相同，稍微向0.5修正1％
     */
    public static List<BigDecimal> calculateNormalization(List<Double> list, boolean ifNeedNormalization) {

        List<BigDecimal> normalization = new ArrayList<>();

        if (ifNeedNormalization) {
            List<Double> extreme = maxAndMinNumInList(list);
            // 极差
            BigDecimal range = minus(extreme.get(0), extreme.get(1));
            for (Double number : list) {
                // 分子
                BigDecimal molecular;
                if (number.equals(extreme.get(1))) {
                    molecular = minus(number*1.01, extreme.get(1));

                } else if (number.equals(extreme.get(0))) {
                    molecular = minus(number*0.99, extreme.get(1));

                } else  {
                    molecular = minus(number, extreme.get(1));
                }

                normalization.add(molecular.divide(range, 10, BigDecimal.ROUND_HALF_UP ));
            }

        } else {
            for (Double value : list) {
                normalization.add(BigDecimal.valueOf(value));
            }
        }

        return normalization;
    }

    /**
     * 计算欧式距离，计算公式 根号((x1-y1)^2+(x2-y2)^2)+...+(xn-yn)^2),即对应差的平方和开根号,相当于向量的长度
     * @param x x
     * @param y y
     * @return 欧式距离
     */
    private static Double calculateEuclideanDistance (List<BigDecimal> x, List<BigDecimal> y) {

        int size = x.size();
        if (size != y.size()) {
            System.out.println("规格化时计算出错");
            return 0.0;
        }
        BigDecimal differenceSquare = BigDecimal.valueOf(0);

        for (int i = 0; i < size; i++) {
            BigDecimal difference = x.get(i).subtract(y.get(i));
            differenceSquare = differenceSquare.add(difference.multiply(difference));
        }

        // 这里使用更为严格的计算，虽然Math.sqrt(double a)调用的也是该方法
        return StrictMath.sqrt(differenceSquare.doubleValue());
    }

    /**
     * 获取极值
     * @param list 需要求极值的数列
     * @return 极值，最大值(index=0)，最小值(index=1)
     */
    public static List<Double> maxAndMinNumInList (List<Double> list) {

        List<Double> result = new ArrayList<>();
        result.add(Collections.max(list));
        result.add(Collections.min(list));
        return result;
    }

    /**
     * 精确加法
     * @param x x
     * @param y y
     * @return 精确加法
     */
    public static BigDecimal plus (Double x, Double y) {

        return BigDecimal.valueOf(x).add(BigDecimal.valueOf(y));
    }

    private static BigDecimal minus (Double x, Double y) {
        return BigDecimal.valueOf(x).subtract(BigDecimal.valueOf(y));
    }

    public BigDecimal multiply (Double x, Double y) {
        return BigDecimal.valueOf(x).multiply(BigDecimal.valueOf(y));
    }

    public static BigDecimal divide (Double x, Double y) {
        return new BigDecimal(x).divide(BigDecimal.valueOf(y), 10, BigDecimal.ROUND_HALF_UP);
    }

    /**
     * 斐波那契数列高效计算
     * @param n 要计算的斐波那契数列的序数
     * @return 对应的斐波那契数列
     */
    public static long fibonacci (long n) {

        if (n == 1) {
            System.out.print(1);
            return 1;

        } else if ( n == 2) {
            System.out.print(1 + ", " + 1);
            return 1;

        }else {
            System.out.print(1 + ", "+ 1 + ", ");
            long m1 = 1L;
            long m2 = 1L;

            for (long i = 3; i < n + 1; i++) {

                // f(n) = f(n-1) + f(n-2)
                m2 = m1 + m2;

                // f(n-1) = f(n) - f(n-2)
                m1 = m2 - m1;

                System.out.print(m2);

                if (i != n) {
                    System.out.print(", ");
                }
            }
            return m2;
        }
    }

    public static double calculateStandardDeviation(List<Double> dataList, Double averageValue) {

        double result = 0.0;

        for (Double value : dataList ) {
            result += calculateSquaredDifference(value, averageValue);
        }

        return result/dataList.size();
    }

    private static double calculateSquaredDifference(Double valueOne, Double valueTwo) {

        double difference = valueOne - valueTwo;
        return difference*difference;
    }

    public static double calculateAverageValue(List<Double> dataList) {

        double result = 0.0;
        for (Double value : dataList) {
            result += value;
        }
        return result/dataList.size();
    }


    /**
     * 正态分布概率 e^-((x-η)^2/2σ^2)/((√2π)σ),这里用n替代η,用o替代σ
     * @param x x
     * @param n 期望
     * @param o 标准差
     * @return 概率
     */
    public static BigDecimal normalDistribution(double x, double n, double o) {

        BigDecimal bigDecimalX = BigDecimal.valueOf(x);
        BigDecimal bigDecimalN = BigDecimal.valueOf(n);
        BigDecimal bigDecimalO = BigDecimal.valueOf(o);

        BigDecimal molecular = bigDecimalX.subtract(bigDecimalN);
        // (x-η)^2
        molecular = molecular.multiply(molecular);

        // 2σ^2
        BigDecimal b = bigDecimalO.multiply(bigDecimalO).multiply(BigDecimal.valueOf(2));

        // ((x-η)^2/2σ^2)
        molecular = molecular.divide(b, 10, BigDecimal. ROUND_HALF_UP);

        // e^-((x-η)^2/2σ^2)
        molecular = BigDecimal.valueOf(StrictMath.pow(Math.E, -molecular.doubleValue()));

        // 分母 (√2π)σ
        BigDecimal denominator = BigDecimal.valueOf(StrictMath.sqrt(2 * Math.PI)).multiply(bigDecimalO);

        return molecular.divide(denominator, 10, BigDecimal.ROUND_HALF_UP);
    }

    /**
     * 打印list
     * @param list 需要打印的list,只能打印泛型为可直接打印出值的list
     */
    public static void printList(List<?> list) {

        for (Object num : list) {
            System.out.print(num + "  ");
        }
        System.out.print('\n');
    }

    /**
     * 检查两个list包含的值是否完全相等，不考虑重复值的情况，顺序可以不一样
     * @param firstList 第一个list
     * @param secondList 第二个list
     * @return true相等，false不相等
     */
    public static boolean ifTwoListSame(List<Integer> firstList, List<Integer> secondList) {

        if (firstList.size() != secondList.size()) {
            return false;
        }

        for (Integer value : firstList) {
            if (!secondList.contains(value)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 根据结果list和对应出现次数返回TreeMap
     * @param values 结果list
     * @param times 对应次数
     * @return 从小到大的TreeMap
     */
    public static TreeMap<Integer, List<Integer>> getListOrder(List<List<Integer>> values, List<Integer> times) {

        TreeMap<Integer, List<Integer>> result = new TreeMap<>();

        for (int index = 0; index < values.size(); index++) {

            result.put(times.get(index), values.get(index));
        }
        return result;
    }

    /**
     * 从list中获取最值的索引，list泛型为Double
     * @param list list
     * @param ifNeedMax 需要的是否是最大的值，true返回最大值的索引, false返回最小值的索引
     * @return 最值索引
     */
    public static <E extends Number> int getMaxValueIndexFromList(List<E> list, boolean ifNeedMax) {

        int index = 0;
        E o = list.get(0);

        if (ifNeedMax) {
            for (int i = 1; i < list.size(); i++) {
                if ((Double)o < (Double) list.get(i)) {
                    index = i;
                    o = list.get(i);
                }
            }

        } else {
            for (int i = 1; i < list.size(); i++) {
                if ((Double)o > (Double) list.get(i)) {
                    index = i;
                    o = list.get(i);
                }
            }
        }

        return index;
    }

    /**
     * 牛顿迭代法计算开根号，精确到收敛
     * 附：计算数与需要的计算次数的变化点，其中的未列出数的计算次数不一定就是已列出的最接近但小于该数的对应的次数，
     * 例如，2428需要计算11次，但大于该数的2430只需要计算10次
     * 1 需要计算1次
     * 2 需要计算5次
     * 4 需要计算6次
     * 12 需要计算7次
     * 46 需要计算8次
     * 183 需要计算9次
     * 607 需要计算10次
     * 2428 需要计算11次
     * 8669 需要计算12次
     * 33478 需要计算13次
     * 124423 需要计算14次
     * 497692 需要计算15次
     * 1744386 需要计算16次
     * 6732259 需要计算17次
     * 26711226 需要计算18次
     * 96172686 需要计算19次
     * @param c 需要计算开根号的值
     * @return java一般精确度下的最精确结果
     */
    private static double sqrt (double c) {
        double value = c;
        double record = 0.0;
        while (true) {
            value = (value + c / value) / 2.0;
            if (record == value) {
                break;
            }
            record = value;
        }
        return value;
    }

    public static void printTwoDimensionArray(double[][] valuesArray) {

        for (double[] values : valuesArray) {
            for (int inIndex = 0; inIndex < valuesArray[0].length; inIndex++) {
                System.out.print(values[inIndex] + "");
            }
            System.out.println("");
        }

    }
}
