package bpNN_iris;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * @ ���ݼ���iris-flower @ ��������
 * @ BP��1����㡢2���ز㡢1�����
 */

public class BpNN2_iris {

	// ѵ��������С
	static int trainNum = 100;
	// ѵ�����ݼ� = ����1-����2-����3-����4-ʵ��ֵ
	static double trainData[][] = new double[trainNum][5];
	// ����������С
	static int testNum = 100;
	// �������ݼ� = ����1-����2-����3-����4-ʵ��ֵ-Ԥ��ֵ
	static double testData[][] = new double[testNum][6];

	// ����㡢���ز㡢����� ��Ԫ����
	static int inputL = 4;
	// ����1
	static int hideL1 = 5;
	// ����2
	static int hideL2 = 3;
	// static int outputL=1;

	// Ŀ�����
	static double error = 0.001;
	// ѧϰ��
	static double lr = 0.1;
	// ��������
	static int epochs = 500;

	// Ȩֵ����
	// �����-���ز�1
	static double W1[][] = new double[hideL1][inputL];
	// ���ز�1-���ز�2
	static double W2[][] = new double[hideL2][hideL1];
	// ���ز�2-�����
	static double W3[] = new double[hideL2];

	// ���ز�1 ��ֵ
	static double Htheta1[] = new double[hideL1];
	// ���ز�2 ��ֵ
	static double Htheta2[] = new double[hideL2];
	// ����� ��ֵ
	static double Otheta = 0;

	// ���
	// ���ز�1
	static double[] hideErr1 = new double[hideL1];
	// ���ز�2
	static double[] hideErr2 = new double[hideL2];
	// �����
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
	 * ����ѵ������
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
			System.out.println("ѵ�����ݼ�����ok!");
			br.close();
		} catch (Exception ex) {
			System.out.println("TrainDataSet Load Failed !");
		}
	}

	/*
	 * ��ʼ��Ȩֵ����
	 */
	public static void initWeight() {
		System.out.println("��ʼ��Ȩֵ����...");
		// ���ز�1-����� W1(i,j)
		for (int i = 0; i < hideL1; i++) {
			for (int j = 0; j < inputL; j++) {
				W1[i][j] = Math.random() / 2;
			}
		}
		// ���ز�2-���ز�1 W2(i,j)
		for (int i = 0; i < hideL2; i++) {
			for (int j = 0; j < hideL1; j++) {
				W2[i][j] = Math.random() / 2;
			}
		}
		// �����-���ز�2 W3(i,j)
		for (int j = 0; j < hideL2; j++) {
			W3[j] = Math.random() / 2;
		}
	}

	/*
	 * ��ʼ����ֵ
	 */
	public static void initTheta() {
		System.out.println("��ʼ����ֵ...");
		// ���ز�1
		for (int i = 0; i < hideL1; i++) {
			Htheta1[i] = Math.random() / 2;
		}
		// ���ز�2
		for (int i = 0; i < hideL2; i++) {
			Htheta2[i] = Math.random() / 2;
		}
		// �����
		Otheta = Math.random() / 2;
	}

	/*
	 * ѵ������
	 */
	public static void trainProcess() {
		// ����ÿ��ѵ������
		for (int i = 0; i < trainNum; i++) {
			// �����-��� = ��������-����ֵ
			double inputO[] = new double[inputL];
			for (int t = 0; t < inputL; t++) {
				inputO[t] = trainData[i][t];
			}

			// ��׼��� = ��������-���
			double outputT = trainData[i][inputL];

			// ��ǰ��������
			// �������ز�1���������������
			double hideI1[] = new double[hideL1];
			double hideO1[] = new double[hideL1];
			for (int ti = 0; ti < hideL1; ti++) {
				hideI1[ti] = addW1ijOj(ti, inputO) + Htheta1[ti];
				hideO1[ti] = sigmoid(hideI1[ti]);
			}

			// �������ز�2���������������
			double hideI2[] = new double[hideL2];
			double hideO2[] = new double[hideL2];
			for (int ti = 0; ti < hideL2; ti++) {
				hideI2[ti] = addW2ijOj(ti, hideO1) + Htheta2[ti];
				hideO2[ti] = sigmoid(hideI2[ti]);
			}

			// ��������㣬�������������
			double outputI = addW3ijOj(hideO2) + Otheta;
			double outputO = sigmoid(outputI);

			// ���򴫲����

			// �������������
			outputErr = -(outputT - outputO) * sigmoid2(outputI);

			// ���ز�2�����
			for (int ti = 0; ti < hideL2; ti++) {
				hideErr2[ti] = outputErr * W3[ti] * sigmoid2(hideI2[ti]);
			}

			// ���ز�1�����
			for (int tj = 0; tj < hideL1; tj++) {
				hideErr1[tj] = addW2jiErrj(tj, hideErr2) * sigmoid2(hideI1[tj]);
			}

			// ����Ȩֵ

			// Ȩϵ�� W3 ����
			for (int tj = 0; tj < hideL2; tj++) {
				double delta = outputErr * hideO2[tj];
				W3[tj] = W3[tj] - lr * delta;
			}

			// ����� ��ֵ����
			double delta1 = outputErr;
			Otheta = Otheta - lr * delta1;

			// Ȩϵ�� W2 ����
			for (int ti = 0; ti < hideL2; ti++) {
				for (int tj = 0; tj < hideL1; tj++) {
					double delta = hideO1[tj] * hideErr2[ti];
					W2[ti][tj] = W2[ti][tj] - lr * delta;
				}
			}

			// ���ز�2 ��ֵ����
			for (int ti = 0; ti < hideL2; ti++) {
				double delta = hideErr2[ti];
				Htheta2[ti] = Htheta2[ti] - lr * delta;
			}

			// Ȩϵ�� W1 ����
			for (int ti = 0; ti < hideL1; ti++) {
				for (int tj = 0; tj < inputL; tj++) {
					double delta = inputO[tj] * hideErr1[ti];
					W1[ti][tj] = W1[ti][tj] - lr * delta;
				}
			}

			// ���ز�1 ��ֵ����
			for (int ti = 0; ti < hideL1; ti++) {
				double delta = hideErr1[ti];
				Htheta1[ti] = Htheta1[ti] - lr * delta;
			}
		}
	}

	/*
	 * ���Թ���
	 */
	public static void testProcess() {
		System.out.println("---- ��ʼ�������ݣ�----");
		// ����ÿ����������
		for (int i = 0; i < testNum; i++) {
			// �����-��� = ��������-����ֵ
			double inputO[] = new double[inputL];
			for (int t = 0; t < inputL; t++) {
				inputO[t] = testData[i][t];
			}

			// ��ǰ��������
			// �������ز�1���������������
			double hideI1[] = new double[hideL1];
			double hideO1[] = new double[hideL1];
			for (int ti = 0; ti < hideL1; ti++) {
				hideI1[ti] = addW1ijOj(ti, inputO) + Htheta1[ti];
				hideO1[ti] = sigmoid(hideI1[ti]);
			}

			// �������ز�2���������������
			double hideI2[] = new double[hideL2];
			double hideO2[] = new double[hideL2];
			for (int ti = 0; ti < hideL2; ti++) {
				hideI2[ti] = addW2ijOj(ti, hideO1) + Htheta2[ti];
				hideO2[ti] = sigmoid(hideI2[ti]);
			}

			// ��������㣬�������������
			double outputI = addW3ijOj(hideO2) + Otheta;
			double outputO = sigmoid(outputI);

			testData[i][5] = outputO;

			// ����������ݼ����
			System.out.print("�������� " + i + " :\t");
			for (int k = 0; k < testData[i].length; k++) {
				System.out.print(testData[i][k] + "\t");
			}
			System.out.println();

		}
	}

	/*
	 * sigmoid����
	 */
	public static double sigmoid(double x) {
		double f = 1.0 / (1 + Math.exp(-1.0 * x));
		return f;
	}

	/*
	 * sigmoid �󵼺���
	 */
	public static double sigmoid2(double x) {
		double f2 = Math.exp(-1.0 * x) / Math.pow(1 + Math.exp(-1 * x), 2);
		return f2;
	}

	/*
	 * �������ز��� W1(i,j)Oj �����
	 */
	public static double addW1ijOj(int i, double[] inputO) {
		double sum = 0;
		for (int j = 0; j < inputL; j++) {
			sum += W1[i][j] * inputO[j];
		}
		return sum;
	}

	/*
	 * �������ز��� W2(i,j)Oj �����
	 */
	public static double addW2ijOj(int i, double[] hideO1) {
		double sum = 0;
		for (int j = 0; j < hideL1; j++) {
			sum += W2[i][j] * hideO1[j];
		}
		return sum;
	}

	/*
	 * ����������� W3(i,j)Oj �����
	 */
	public static double addW3ijOj(double[] hideO2) {
		double sum = 0;
		for (int j = 0; j < hideL2; j++)
			sum += W3[j] * hideO2[j];
		return sum;
	}

	/*
	 * �������ز�2�� W2(i,j)Errj �����
	 */
	public static double addW2jiErrj(int j, double[] hideErr2) {
		double sum = 0;
		for (int i = 0; i < hideL2; i++) {
			sum += W2[i][j] * hideErr2[i];
		}
		return sum;
	}

}
