package bpNN_iris;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * @ 数据集：iris-flower @ 两类问题
 * @ BP：1输入层、1隐藏层、1输出层
 */

public class BpNN_iris {

	// 训练样本大小
	static int trainNum = 100;
	// 训练数据集 = 属性1-属性2-属性3-属性4-实际值
	static double trainData[][] = new double[trainNum][5];
	// 测试样本大小
	static int testNum = 100;
	// 测试数据集 = 属性1-属性2-属性3-属性4-实际值-预测值
	static double testData[][] = new double[testNum][6];

	// 输入层、隐藏层、输出层 神经元个数
	static int inputL = 4;
	static int hideL = 10;
	// static int outputL=1;

	// 目标误差
	static double error = 0.001;
	// 学习率
	static double lr = 0.1;
	// 迭代次数
	static int epochs = 500;

	// 权值矩阵
	// 输入层-隐藏层
	static double W1[][] = new double[hideL][inputL];
	// 隐藏层-输出层
	static double W2[] = new double[hideL];
	// 阈值
	static double Htheta[] = new double[hideL];
	static double Otheta = 0;

	// 误差
	// 隐藏层
	static double[] hideErr = new double[hideL];
	// 输出层
	static double outputErr = 0;

	public static void main(String[] args) {
		loadTrainData();
		initWeight();
		initTheta();
		for (int e = 0; e < epochs; e++) {
			// System.out.print("Epoch:"+e);
			trainProcess();
			// System.out.println();
		}
		testProcess();
	}

	/**
	 * 导入训练数据
	 */
	public static void loadTrainData() {
		File file = new File("datasets\\Iris_data2.txt");
		FileReader fr = null;
		try {
			fr = new FileReader(file.getAbsolutePath());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		BufferedReader br = new BufferedReader(fr);
		int n = 0;
		try {
			String line = null;
			while ((line = br.readLine()) != null) {
				String data[] = line.split("\t");
				for (int i = 0; i < data.length; i++) {
					testData[n][i] = Double.parseDouble(data[i]);
					trainData[n][i] = Double.parseDouble(data[i]);
				}
				n++;
			}
			System.out.println("训练数据集导入ok!");
			br.close();
		} catch (Exception ex) {
			System.out.println("TrainDataSet Load Failed !");
		}
	}

	/*
	 * 初始化权值矩阵
	 */
	public static void initWeight() {
		System.out.println("初始化权值矩阵...");
		// 隐藏层-输入层 W1(i,j)
		for (int i = 0; i < hideL; i++) {
			for (int j = 0; j < inputL; j++) {
				W1[i][j] = Math.random() / 2;
			}
		}
		// 输出层-隐藏层 W2(i,j)
		for (int j = 0; j < hideL; j++) {
			W2[j] = Math.random() / 2;
		}
	}

	/*
	 * 初始化阈值
	 */
	public static void initTheta() {
		System.out.println("初始化阈值...");
		// 隐藏层
		for (int i = 0; i < hideL; i++) {
			Htheta[i] = Math.random() / 2;
		}
		// 输出层
		Otheta = Math.random() / 2;
	}

	/*
	 * 训练过程
	 */
	public static void trainProcess() {
		// 对于每个训练样本
		for (int i = 0; i < trainNum; i++) {
			// 输入层-输出 = 样本数据-属性值
			double inputO[] = new double[inputL];
			for (int t = 0; t < inputL; t++) {
				inputO[t] = trainData[i][t];
			}

			// 标准输出 = 样本数据-类别
			double outputT = trainData[i][inputL];

			// 向前传播输入
			// 对于隐藏层，计算输入
			double hideI[] = new double[hideL];
			double hideO[] = new double[hideL];
			for (int ti = 0; ti < hideL; ti++) {
				hideI[ti] = addW1ijOj(ti, inputO) + Htheta[ti];
				hideO[ti] = sigmoid(hideI[ti]);
			}

			// 对于输出层，计算输入与输出
			double outputI = addW2ijOj(hideO) + Otheta;
			double outputO = sigmoid(outputI);

			// 后向传播误差
			// 计算输出层的误差
			outputErr = -(outputT - outputO) * sigmoid2(outputI);

			// 隐藏层的误差
			for (int ti = 0; ti < hideL; ti++) {
				hideErr[ti] = outputErr * W2[ti] * sigmoid2(hideI[ti]);
			}

			// 权系数 W2 更新
			for (int tj = 0; tj < hideL; tj++) {
				double delta = outputErr * hideO[tj];
				W2[tj] = W2[tj] - lr * delta;
			}

			// 输出层 阈值更新
			double delta1 = outputErr;
			Otheta = Otheta - lr * delta1;

			// 权系数 W1 更新
			for (int ti = 0; ti < hideL; ti++) {
				for (int tj = 0; tj < inputL; tj++) {
					double delta = inputO[tj] * hideErr[ti];
					W1[ti][tj] = W1[ti][tj] - lr * delta;
				}
			}

			// 隐藏层 阈值更新
			for (int ti = 0; ti < hideL; ti++) {
				double delta2 = hideErr[ti];
				Htheta[ti] = Htheta[ti] - lr * delta2;
			}
		}
	}

	/*
	 * 测试过程
	 */
	public static void testProcess() {
		System.out.println("--- 开始测试数据：----");
		// 对于每个测试样本
		for (int i = 0; i < testNum; i++) {
			// 输入层-输出 = 样本数据-属性值
			double inputO[] = new double[inputL];
			for (int t = 0; t < inputL; t++) {
				inputO[t] = testData[i][t];
			}

			// 向前传播输入
			// 对于隐藏层，计算输入
			double hideI[] = new double[hideL];
			double hideO[] = new double[hideL];
			for (int ti = 0; ti < hideL; ti++) {
				hideI[ti] = addW1ijOj(ti, inputO) + Htheta[ti];
				hideO[ti] = sigmoid(hideI[ti]);
			}

			// 对于输出层，计算输入与输出
			double outputI = addW2ijOj(hideO) + Otheta;
			double outputO = sigmoid(outputI);

			testData[i][5] = outputO;

			// 输出测试数据及结果
			System.out.print("测试样本 " + i + " :\t");
			for (int k = 0; k < testData[i].length; k++) {
				System.out.print(testData[i][k] + "\t");
			}
			System.out.println();
		}

	}

	/*
	 * sigmoid函数
	 */
	public static double sigmoid(double x) {
		double f = 1.0 / (1 + Math.exp(-1.0 * x));
		return f;
	}

	/*
	 * sigmoid 求导函数
	 */
	public static double sigmoid2(double x) {
		double f2 = Math.exp(-1.0 * x) / Math.pow(1 + Math.exp(-1 * x), 2);
		return f2;
	}

	/*
	 * 计算隐藏层中 W1(i,j)Oj 的求和
	 */
	public static double addW1ijOj(int i, double[] inputO) {
		double sum = 0;
		for (int j = 0; j < inputL; j++) {
			sum += W1[i][j] * inputO[j];
		}
		return sum;
	}

	/*
	 * 计算输出层中 W2(i,j)Oj 的求和
	 */
	public static double addW2ijOj(double[] hideO) {
		double sum = 0;
		for (int j = 0; j < hideL; j++)
			sum += W2[j] * hideO[j];
		return sum;
	}

}
