package algorithm.gradient.descent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/7/8
 */
class BaseDataConstructForRegression extends BaseDataConstruct {

    /**
     * 是否需要添加平方项
     */
    boolean ifNeedSquare;
    /**
     * 是否需要添加两两参数相乘项
     */
    boolean ifNeedTwoParamMultiply;

    /**
     * 给参数数组添加数据
     * @param dataParamsList 参数数组
     * @param values 一条数据
     */
    void addDataToDataParamsList(List<List<Double>> dataParamsList, Double...  values) {

        List<Double> list = new ArrayList<>();
        Collections.addAll(list, values);

        if (ifNeedSquare) {
            for (Double value : values) {
                list.add(value*value);
            }
        }

        if (ifNeedTwoParamMultiply) {
            addArbitraryTwoParamMultiply(list, values.length);
        }

        dataParamsList.add(list);
    }

    /**
     * 添加参数两两相乘项
     * @param list 需要添加参数的列表
     * @param originalSize 原始参数数量
     */
    private void addArbitraryTwoParamMultiply(List<Double> list, int originalSize) {

        for (int firstIndex = 0; firstIndex < originalSize; firstIndex++) {

            for (int secondIndex = firstIndex+1; secondIndex < originalSize; secondIndex++) {

                list.add(list.get(firstIndex)*list.get(secondIndex));
            }
        }
    }
}
