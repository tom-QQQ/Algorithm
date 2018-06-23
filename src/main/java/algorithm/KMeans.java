package algorithm;

import utils.Utils;

import java.math.BigDecimal;
import java.util.*;

/**
 * @author Ning
 * @date Create in 2018/5/3
 */
public class KMeans {

    public static void calculateExampleResult() {

        List<List<Integer>> results = new ArrayList<>();
        List<Integer> times = new ArrayList<>();
        Integer errorTimes = 0;

        for (int i = 0; i < 10000; i++) {
            List<List<Integer>> result = kMeans();

            if (result == null) {
                errorTimes++;
                continue;
            }

            for (List<Integer> values : result) {

                boolean ifTwoListSame = false;
                for (int index = 0; index < results.size(); index ++) {

                    ifTwoListSame = Utils.ifTwoListSame(values,results.get(index));
                    if (ifTwoListSame) {
                        times.set(index, times.get(index)+1);
                        break;
                    }
                }

                if (!ifTwoListSame) {
                    results.add(values);
                    times.add(1);
                }
            }
        }

        System.out.println("出错" + errorTimes + "次");


        List<List<Integer>> finalResult = new ArrayList<>();

        while (true) {

            // 查询出出现次数最高对应的索引
            int maxIndex = 0;
            Integer maxValue = times.get(0);
            for (int index = 1; index < times.size(); index++) {

                if (maxValue < times.get(index)) {
                    maxIndex = index;
                    maxValue = times.get(index);
                }

            }

            // 最终结果值，用于对比
            List<Integer> resultValue = results.get(maxIndex);
            finalResult.add(resultValue);

            for (int index = results.size() -1; index >= 0; index--) {

                // 如果其他值中包含了最终结果的任意一个值，则剔除
                for (Integer value : resultValue) {
                    if (results.get(index).contains(value)) {
                        results.remove(index);
                        times.remove(index);
                        break;
                    }
                }
            }

            if (results.size() == 0 && results.size() == times.size()) {
                break;
            }
        }

        System.out.println("最终聚类结果：");
        System.out.println(finalResult);
    }

    private static List<List<Integer>> kMeans () {

        List<Double> worldCup2006 = new ArrayList<>();
        worldCup2006.add(50.0);
        worldCup2006.add(28.0);
        worldCup2006.add(17.0);
        worldCup2006.add(25.0);
        worldCup2006.add(28.0);
        worldCup2006.add(50.0);
        worldCup2006.add(50.0);
        worldCup2006.add(50.0);
        worldCup2006.add(40.0);
        worldCup2006.add(50.0);
        worldCup2006.add(50.0);
        worldCup2006.add(50.0);
        worldCup2006.add(40.0);
        worldCup2006.add(40.0);
        worldCup2006.add(50.0);
        List<BigDecimal> normalizationWorldCup2006 = Utils.calculateNormalization(worldCup2006, true);

        List<Double> worldCup2010 = new ArrayList<>();
        worldCup2010.add(50.0);
        worldCup2010.add(9.0);
        worldCup2010.add(15.0);
        worldCup2010.add(40.0);
        worldCup2010.add(40.0);
        worldCup2010.add(50.0);
        worldCup2010.add(40.0);
        worldCup2010.add(40.0);
        worldCup2010.add(40.0);
        worldCup2010.add(50.0);
        worldCup2010.add(50.0);
        worldCup2010.add(50.0);
        worldCup2010.add(40.0);
        worldCup2010.add(32.0);
        worldCup2010.add(50.0);
        List<BigDecimal> normalizationWorldCup2010 = Utils.calculateNormalization(worldCup2010, true);


        List<Double> asiaCup2007 = new ArrayList<>();
        asiaCup2007.add(9.0);
        asiaCup2007.add(4.0);
        asiaCup2007.add(3.0);
        asiaCup2007.add(5.0);
        asiaCup2007.add(2.0);
        asiaCup2007.add(1.0);
        asiaCup2007.add(9.0);
        asiaCup2007.add(9.0);
        asiaCup2007.add(5.0);
        asiaCup2007.add(9.0);
        asiaCup2007.add(5.0);
        asiaCup2007.add(9.0);
        asiaCup2007.add(9.0);
        asiaCup2007.add(17.0);
        asiaCup2007.add(9.0);
        List<BigDecimal> normalizationAsiaCup2007 = Utils.calculateNormalization(asiaCup2007, true);


        if (worldCup2010.size() != worldCup2006.size() || worldCup2010.size() != asiaCup2007.size()) {
            System.out.println("数据输入有误，不能计算");
            return null;
        }


        // 开始选种子
        Random random = new Random();
        // 随机选取其中一个作为第一个种子
        int seedIndex = random.nextInt(15);
//        System.out.println("第1个选取的种子索引为:" + seedIndex);


        // 这里用TreeMap避免后期取值时的排序
        TreeMap<Integer, List<Double>> originalSeeds = new TreeMap<>();

        List<Double> seed = new ArrayList<>();
        seed.add(normalizationWorldCup2010.get(seedIndex).doubleValue());
        seed.add(normalizationWorldCup2006.get(seedIndex).doubleValue());
        seed.add(normalizationAsiaCup2007.get(seedIndex).doubleValue());
        originalSeeds.put(seedIndex, seed);


        for (int i = 1; i < 3; i++) {

            // 已选种子的索引
            Integer[] seedIndexes = new Integer[originalSeeds.size()];
            // 取出已有种子索引
            originalSeeds.keySet().toArray(seedIndexes);


            // 种子点到 剩余 备选种子点的欧氏距离
            List<Double> otherDistance = new ArrayList<>();
            for (int j = 0; j < worldCup2006.size(); j++) {

                List<Double> otherSeed = new ArrayList<>();

                // 排除已选择的种子
                boolean ifThisSeedHasSelected = false;
                for (Integer num : seedIndexes) {
                    if (j == num) {
                        ifThisSeedHasSelected = true;
                        break;
                    }
                }

                if (!ifThisSeedHasSelected) {
                    otherSeed.add(normalizationWorldCup2006.get(j).doubleValue());
                    otherSeed.add(normalizationWorldCup2010.get(j).doubleValue());
                    otherSeed.add(normalizationAsiaCup2007.get(j).doubleValue());
                    otherDistance.add(Utils.normalizationEuclideanDistance(seed, otherSeed, false));
                }
            }

            List<Double> maxAndMin = Utils.maxAndMinNumInList(otherDistance);
            Double range = maxAndMin.get(0) - maxAndMin.get(1);
            int seedTempIndex;
            while (true) {
                seedTempIndex = random.nextInt(14 - i);
                // 位于otherDistance的最小值到最大值之间的随机数 = [0,1]的随机double*(最大值 - 最小值) + 最小值
                double randomValue = random.nextDouble() * range + maxAndMin.get(1);

                if (randomValue - otherDistance.get(seedTempIndex) <= 0) {

                    List<Integer> orderIndex = Arrays.asList(seedIndexes);
                    // 由小到大排列
                    orderIndex.sort((o1, o2) -> (o1 - o2));
                    for (Integer num : orderIndex) {
                        // 临时索引每大于一个正确索引，临时索引值就+1
                        if (seedTempIndex >= num) {
                            seedTempIndex++;
                        } else {
                            break;
                        }
                    }

//                    System.out.println("第" + (i + 1) + "个种子的索引为: " + seedTempIndex);
                    // 将参照种子修改为新获取到的种子
                    seed.clear();
                    seed.add(normalizationWorldCup2010.get(seedTempIndex).doubleValue());
                    seed.add(normalizationWorldCup2006.get(seedTempIndex).doubleValue());
                    seed.add(normalizationAsiaCup2007.get(seedTempIndex).doubleValue());
                    break;
                }
            }

            List<Double> otherSeed = new ArrayList<>();
            otherSeed.add(normalizationWorldCup2006.get(seedTempIndex).doubleValue());
            otherSeed.add(normalizationWorldCup2010.get(seedTempIndex).doubleValue());
            otherSeed.add(normalizationAsiaCup2007.get(seedTempIndex).doubleValue());
            originalSeeds.put(seedTempIndex, otherSeed);

        }


        // 已选种子的索引,由于该值从TreeMap中取出，所以顺序是从小到大
        Integer[] seedIndexes = new Integer[originalSeeds.size()];
        // 取出已有种子索引
        originalSeeds.keySet().toArray(seedIndexes);

        List<List<Integer>> clusters = saveOriginalSeeds(seedIndexes);
        List<List<Double>> seedsValue = saveOriginalSeedsValue(originalSeeds, seedIndexes);

        for (int i = 0; i < asiaCup2007.size(); i++) {

            // 排除已选择的种子
            boolean ifThisSeedHasSelected = false;
            // 循环值大于等于最小值，或小于等于最大值时才循环,即范围位于最大值和最小值之间时
            if (i >= seedIndexes[0] || i <= seedIndexes[seedIndexes.length - 1]) {
                for (Integer num : seedIndexes) {
                    if (i == num) {
                        ifThisSeedHasSelected = true;
                        break;
                    }
                }
            }

            if (!ifThisSeedHasSelected) {
                List<Double> distanceToOriginalSeeds = new ArrayList<>();

                for (List<Double> seeds : seedsValue) {

                    List<Double> optionalValue = new ArrayList<>();
                    optionalValue.add(normalizationWorldCup2006.get(i).doubleValue());
                    optionalValue.add(normalizationWorldCup2010.get(i).doubleValue());
                    optionalValue.add(normalizationAsiaCup2007.get(i).doubleValue());

                    distanceToOriginalSeeds.add(Utils.normalizationEuclideanDistance(seeds, optionalValue, false));
                }

                int belongSeedIndex = Utils.getMaxValueIndexFromList(distanceToOriginalSeeds, true);
                clusters.get(belongSeedIndex).add(i);
            }
        }

//        Utils.printList(clusters);

//        int calculateTimes = 0;
        while (true) {

            List<List<Double>> refValue = new ArrayList<>();
            for (List<Integer> cluster : clusters) {
                // 用于存储平均值，方便起见先存储各个项的值
                List<Double> averageValue = new ArrayList<>(cluster.size() * 4);

                for (Integer value : cluster) {
                    // 先存储各个项的值
                    averageValue.add(normalizationWorldCup2006.get(value).doubleValue());
                    averageValue.add(normalizationWorldCup2010.get(value).doubleValue());
                    averageValue.add(normalizationAsiaCup2007.get(value).doubleValue());
                }

                Double averageWorldCup2006 = 0.00;
                Double averageWorldCup2010 = 0.00;
                Double averageAsiaCup2007 = 0.00;
                // 根据之前的和计算出平均值并修改原值
                for (int valueIndex = 0; valueIndex < cluster.size(); valueIndex++) {
                    averageWorldCup2006 = Utils.plus(averageWorldCup2006, averageValue.get(valueIndex * 3)).doubleValue();
                    averageWorldCup2010 = Utils.plus(averageWorldCup2010, averageValue.get(valueIndex * 3 + 1)).doubleValue();
                    averageAsiaCup2007 = Utils.plus(averageAsiaCup2007, averageValue.get(valueIndex * 3 + 2)).doubleValue();
                }

                averageValue.clear();
                if (cluster.size() == 0) {
                    return null;
                }
                averageValue.add(Utils.divide(averageWorldCup2006, (double) cluster.size()).doubleValue());
                averageValue.add(Utils.divide(averageWorldCup2010, (double) cluster.size()).doubleValue());
                averageValue.add(Utils.divide(averageAsiaCup2007, (double) cluster.size()).doubleValue());

                refValue.add(averageValue);
            }

            List<List<Integer>> newClusters = new ArrayList<>();
            newClusters.add(new ArrayList<>());
            newClusters.add(new ArrayList<>());
            newClusters.add(new ArrayList<>());

            for (int index = 0; index < asiaCup2007.size(); index++) {

                List<Double> distanceToOriginalSeeds = new ArrayList<>();

                for (List<Double> values : refValue) {
                    List<Double> optionalValue = new ArrayList<>();
                    optionalValue.add(normalizationWorldCup2006.get(index).doubleValue());
                    optionalValue.add(normalizationWorldCup2010.get(index).doubleValue());
                    optionalValue.add(normalizationAsiaCup2007.get(index).doubleValue());
                    distanceToOriginalSeeds.add(Utils.normalizationEuclideanDistance(values, optionalValue, false));
                }

                int belongSeedIndex = Utils.getMaxValueIndexFromList(distanceToOriginalSeeds, false);
                newClusters.get(belongSeedIndex).add(index);
            }

            boolean ifSame = false;
            for (int index = 0; index < newClusters.size() - 1; index++) {

                ifSame = Utils.ifTwoListSame(clusters.get(index), newClusters.get(index));
            }

            // 判断完新计算出的list和之前的list是否相同后将新的list赋值给之前的list
            copySecondListToFirstList(clusters, newClusters);

            if (ifSame) {
                break;
            }
//            calculateTimes++;
        }

//        System.out.println("最终结果为: ");
//        ning.utils.Utils.printList(clusters);
//        System.out.println("计算了" + calculateTimes + "次");

        return clusters;
    }

    /**
     * 存储之前得到的几个原始种子的索引
     * @return 簇集合
     */
    private static List<List<Integer>> saveOriginalSeeds (Integer[] seedIndexes) {

        // 簇
        List<List<Integer>> clusters = new ArrayList<>();

        for (Integer num : seedIndexes) {
            List<Integer> cluster = new ArrayList<>();
            cluster.add(num);
            clusters.add(cluster);
        }
        return clusters;
    }

    /**
     * 返回原始种子的值
     * @param originalSeeds 原始种子
     * @param seedIndexes 原始种子索引
     * @return 原始种子值
     */
    private static List<List<Double>> saveOriginalSeedsValue(TreeMap<Integer, List<Double>> originalSeeds, Integer[] seedIndexes) {

        List<List<Double>> originalSeedsValue = new ArrayList<>();
        for (Integer num : seedIndexes) {
            originalSeedsValue.add(originalSeeds.get(num));
        }
        return originalSeedsValue;
    }


    /**
     * 将第二个list的值赋值给第一个
     * @param firstList 第一个list
     * @param secondList 第二个list
     */
    private static void copySecondListToFirstList (List<List<Integer>> firstList, List<List<Integer>> secondList) {

        firstList.clear();

        for (List<Integer> list : secondList) {

            List<Integer> values = new ArrayList<>(list);
            firstList.add(values);
        }
    }
}
