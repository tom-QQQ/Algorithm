import algorithm.gradient.descent.LinearRegression;
import algorithm.gradient.descent.LogisticRegression;

/**
 * @author Ning
 * create on 2018/04/10
 */
public class Main {

    public static void main(String[] args) {

//        KMeans.calculateExampleResult();

//        LinearRegression regression = new LinearRegression(false, false);

//        regression.calculateExampleResult();

//        regression.verificationResult();

        LogisticRegression logisticRegression = new LogisticRegression(true, true);
        logisticRegression.calculateExampleResult();
//
//        logisticRegression.verificationResult();

    }
}







