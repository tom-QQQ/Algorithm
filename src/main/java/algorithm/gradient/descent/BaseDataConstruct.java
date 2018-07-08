package algorithm.gradient.descent;

import org.ujmp.core.Matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Ning
 * @date Create in 2018/6/24
 */
class BaseDataConstruct extends BaseExamine {

    int initialNumberRange;
    double studyRate;
    double declineValue;
    long maxLoop;
    double convergence;

    /**
     * 正则系数，为0时不进行正则化,当正则系数较小但仍然大于0时，需要迭代计算更多次，但更容易发现相关性较小的参数，但再向0接近时，迭代次数又会减少
     * 如果正则系数较大，则代价也会增加很多，推荐范围[0.005, 1]，该值较大时，即使去除代价的正则部分，代价依然很大，此时需要更多的参数，如平方项等
     */
    double lambda;

    {
        initialNumberRange = 40;
        studyRate = 1.0;
        declineValue = 0.618;
        maxLoop = 1000L;
        convergence = 0.0001;
        lambda = 1.0;
    }

    /**
     * 生成指定范围内的随机值
     * @return 指定范围的随机值
     */
    double getNumberInSpecificRange() {

        double number = Math.random()*initialNumberRange;

        boolean ifNegative = Math.random() < 0.5;

        return (ifNegative ? -number : number);
    }

    /**
     * 根据结果数据创建对应矩阵，该方法无需在子类中调用
     * @param dataResults 结果数据
     * @return 对应矩阵
     */
    Matrix createMatrixWithList(List<Double> dataResults) {

        int resultSize = dataResults.size();

        Matrix matrix = Matrix.Factory.zeros(resultSize, 1);

        for (int valueIndex = 0; valueIndex < resultSize; valueIndex++) {
            matrix.setAsDouble(dataResults.get(valueIndex), valueIndex, 0);
        }

        return matrix;
    }
}
