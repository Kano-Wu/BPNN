package bpNN_iris;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * @ ���ݼ���iris-flower @ ��������
 * @ BP��1����㡢1���ز㡢1�����
 */

public class BpNN_iris {

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
	static int hideL = 10;
	// static int outputL=1;

	// Ŀ�����
	static double error = 0.001;
	// ѧϰ��
	static double lr = 0.1;
	// ��������
	static int epochs = 500;

	// Ȩֵ����
	// �����-���ز�
	static double W1[][] = new double[hideL][inputL];
	// ���ز�-�����
	static double W2[] = new double[hideL];
	// ��ֵ
	static double Htheta[] = new double[hideL];
	static double Otheta = 0;

	// ���
	// ���ز�
	static double[] hideErr = new double[hideL];
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
		// ���ز�-����� W1(i,j)
		for (int i = 0; i < hideL; i++) {
			for (int j = 0; j < inputL; j++) {
				W1[i][j] = Math.random() / 2;
			}
		}
		// �����-���ز� W2(i,j)
		for (int j = 0; j < hideL; j++) {
			W2[j] = Math.random() / 2;
		}
	}

	/*
	 * ��ʼ����ֵ
	 */
	public static void initTheta() {
		System.out.println("��ʼ����ֵ...");
		// ���ز�
		for (int i = 0; i < hideL; i++) {
			Htheta[i] = Math.random() / 2;
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
			// �������ز㣬��������
			double hideI[] = new double[hideL];
			double hideO[] = new double[hideL];
			for (int ti = 0; ti < hideL; ti++) {
				hideI[ti] = addW1ijOj(ti, inputO) + Htheta[ti];
				hideO[ti] = sigmoid(hideI[ti]);
			}

			// ��������㣬�������������
			double outputI = addW2ijOj(hideO) + Otheta;
			double outputO = sigmoid(outputI);

			// ���򴫲����
			// �������������
			outputErr = -(outputT - outputO) * sigmoid2(outputI);

			// ���ز�����
			for (int ti = 0; ti < hideL; ti++) {
				hideErr[ti] = outputErr * W2[ti] * sigmoid2(hideI[ti]);
			}

			// Ȩϵ�� W2 ����
			for (int tj = 0; tj < hideL; tj++) {
				double delta = outputErr * hideO[tj];
				W2[tj] = W2[tj] - lr * delta;
			}

			// ����� ��ֵ����
			double delta1 = outputErr;
			Otheta = Otheta - lr * delta1;

			// Ȩϵ�� W1 ����
			for (int ti = 0; ti < hideL; ti++) {
				for (int tj = 0; tj < inputL; tj++) {
					double delta = inputO[tj] * hideErr[ti];
					W1[ti][tj] = W1[ti][tj] - lr * delta;
				}
			}

			// ���ز� ��ֵ����
			for (int ti = 0; ti < hideL; ti++) {
				double delta2 = hideErr[ti];
				Htheta[ti] = Htheta[ti] - lr * delta2;
			}
		}
	}

	/*
	 * ���Թ���
	 */
	public static void testProcess() {
		System.out.println("--- ��ʼ�������ݣ�----");
		// ����ÿ����������
		for (int i = 0; i < testNum; i++) {
			// �����-��� = ��������-����ֵ
			double inputO[] = new double[inputL];
			for (int t = 0; t < inputL; t++) {
				inputO[t] = testData[i][t];
			}

			// ��ǰ��������
			// �������ز㣬��������
			double hideI[] = new double[hideL];
			double hideO[] = new double[hideL];
			for (int ti = 0; ti < hideL; ti++) {
				hideI[ti] = addW1ijOj(ti, inputO) + Htheta[ti];
				hideO[ti] = sigmoid(hideI[ti]);
			}

			// ��������㣬�������������
			double outputI = addW2ijOj(hideO) + Otheta;
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
	 * ����������� W2(i,j)Oj �����
	 */
	public static double addW2ijOj(double[] hideO) {
		double sum = 0;
		for (int j = 0; j < hideL; j++)
			sum += W2[j] * hideO[j];
		return sum;
	}

}
