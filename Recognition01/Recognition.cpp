#include <iostream>
#include <fstream>
#include <vector>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

using namespace std;

#define LEARN 50 //Epoch
#define SAMPLE 60000 //Number of Training Data Set
#define SAMPLE2 10000 //Number of Test Data Set
#define INPUT 784 //Number of Input Layor Unit
#define HIDDEN 625 //Number of Hidden Layor Unit
#define OUTPUT 10 //Number of Output Layor Unit
#define ALPHA 0.001 // Learning Rate

double x[INPUT + 1], a[HIDDEN + 1], y[OUTPUT]; //x:Input a:Hidden y:Output //x[@+1],a[@+1] => bias or threshold
double w1[INPUT + 1][HIDDEN], w2[HIDDEN + 1][OUTPUT]; //w1:Weight(Input-Hidden), w2:Weight(Hidden-Output) //w1[@+1][],w2[@+1][] => bias or threshold
double a_back[HIDDEN + 1], delta[OUTPUT] = {}; //for Backward Propagation

int err_cnt;

vector<vector<int> > tr_images, te_images; //tr_images:Training Image Set, te_images:Test Image Set
vector<int> tr_label, te_label; //tr_label:Training Label Set, te_label:Test Label Set

/* Sigmoid Function */
double sigmoid(double x){
	double f;
	f = 1.0 / (1.0 + exp(-x));
	return f;
}


/* MNIST Class */
class Mnist{
public:
	vector<vector<int> > readMnistData(string filename, string trte);
	vector<int> readMnistLabel(string filename, string trte);
};

/* for Reading MNIST */
int convert(int x){
	unsigned char c1, c2, c3, c4;

	c1 = x & 255;
	c2 = (x >> 8) & 255;
	c3 = (x >> 16) & 255;
	c4 = (x >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


vector<vector<int> > Mnist::readMnistData(string filename, string trte){
	ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
	int magic_num = 0;
	int img_num = 0; //Number of Images
	int rows = 0;
	int cols = 0;

	/* Acquisition of Header Information (16byte) */
	ifs.read((char*)&magic_num, sizeof(magic_num));
	magic_num = convert(magic_num);
	ifs.read((char*)&img_num, sizeof(img_num));
	img_num = convert(img_num);
	ifs.read((char*)&rows, sizeof(rows));
	rows = convert(rows);
	ifs.read((char*)&cols, sizeof(cols));
	cols = convert(cols);

	if (trte == "tr") {
		tr_images.resize(img_num);

		for (int i = 0; i < img_num; i++){
			tr_images[i].resize(rows * cols);

			for (int row = 0; row < rows; row++){
				for (int col = 0; col < cols; col++){
					unsigned char temp = 0;
					ifs.read((char*)&temp, sizeof(temp));
					tr_images[i][rows*row + col] = (temp > 127 ? 1 : 0); /* Image Thresholding (0=0, 255=1) */
				}
			}
		}

		return tr_images;
	}
	else {
		te_images.resize(img_num);

		for (int i = 0; i < img_num; i++){
			te_images[i].resize(rows * cols);

			for (int row = 0; row < rows; row++){
				for (int col = 0; col < cols; col++){
					unsigned char temp = 0;
					ifs.read((char*)&temp, sizeof(temp));
					te_images[i][rows*row + col] = (temp > 127 ? 1 : 0); /* Image Thresholding (0=0, 255=1) */
				}
			}
		}

		return te_images;
	}
}

vector<int> Mnist::readMnistLabel(string filename, string trte){
	ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
	int magic_num = 0;
	int img_num = 0;

	/* Acquisition of Header Information */
	ifs.read((char*)&magic_num, sizeof(magic_num));
	magic_num = convert(magic_num);
	ifs.read((char*)&img_num, sizeof(img_num));
	img_num = convert(img_num);

	if (trte == "tr") {
		tr_label.resize(img_num);

		cout << "Training Data Set: " << img_num << endl;

		for (int i = 0; i < img_num; i++){
			unsigned char temp = 0;
			ifs.read((char*)&temp, sizeof(temp));
			tr_label[i] = temp;
		}

		return tr_label;
	}
	else {
		te_label.resize(img_num);

		cout << "Test Data Set: " << img_num << endl;

		for (int i = 0; i < img_num; i++){
			unsigned char temp = 0;
			ifs.read((char*)&temp, sizeof(temp));
			te_label[i] = temp;
		}

		return te_label;
	}
}

/* Set Weight */
void setWeight1(double wx[INPUT + 1][HIDDEN]) {
	srand((unsigned)time(NULL));
	for (int i = 0; i < INPUT + 1; i++){
		for (int j = 0; j < HIDDEN; j++){
			wx[i][j] = (double)rand() / (RAND_MAX + 1.0) / 100.0;
		}
	}
}

/* Set Weight */
void setWeight2(double wx[HIDDEN + 1][OUTPUT]) {
	srand((unsigned)time(NULL));
	for (int i = 0; i < HIDDEN + 1; i++){
		for (int j = 0; j < OUTPUT; j++){
			wx[i][j] = (double)rand() / (RAND_MAX + 1.0) / 100.0;
		}
	}
}

/* Training */
void train(int ilearn, double lsum, double error[SAMPLE], double errsum[LEARN]) {
	int temp;
	double alp = (float)ALPHA;
	double z1, target;
	err_cnt = 0;

	for (int isample = 0; isample < SAMPLE; isample++){

		/************************ FORWARD PROPAGATION ***********************/

		/* Set the Inputs */
		for (int i = 0; i < INPUT; i++){
			x[i] = tr_images[isample][i];
		}

		/* Threshold */
		x[INPUT] = (double)1.0;

		/* Calculate the Hiddens(a) */
		for (int j = 0; j < HIDDEN; j++){
			z1 = 0;
			for (int i = 0; i < INPUT + 1; i++){
				z1 = z1 + w1[i][j] * x[i];
			}
			a[j] = sigmoid(z1); /* Applying to Sigmoid Func */
		}

		/* Threshold */
		a[HIDDEN] = (double)1.0;


		/* Calculate the Outputs */
		for (int j = 0; j < OUTPUT; j++){
			z1 = 0;
			for (int i = 0; i < HIDDEN + 1; i++){
				z1 = z1 + w2[i][j] * a[i];
			}
			y[j] = sigmoid(z1); /* Applying to Sigmoid Func */
		}


		/************************ BACK PROPAGATION ***********************/

		/* Outputs (back) */
		for (int j = 0; j < OUTPUT; j++){
			if (j == tr_label[isample]){ target = 1.0; }
			else { target = 0.0; }
			delta[j] = -((double)target - y[j])*((double)1.0 - y[j]) * y[j]; /* Calculate the error(delta) */
			lsum = lsum + delta[j];
			error[isample] += (target - y[j])*(target - y[j]); /* Calculate the error(for log) */
		}

		temp = 0;
		for (int j = 0; j < OUTPUT; j++){
			if (y[temp] < y[j]){
				temp = j;
			}
		}

		lsum = lsum / (OUTPUT*1.0);

		/* Hiddens (back) => delta? */
		for (int i = 0; i < HIDDEN; i++){
			z1 = 0;
			for (int j = 0; j < OUTPUT; j++){
				z1 = z1 + w2[i][j] * delta[j];
			}
			a_back[i] = z1 * ((double)1.0 - a[i]) * a[i];
		}

		/* Weight Updates */
		for (int i = 0; i < INPUT + 1; i++){
			for (int j = 0; j < HIDDEN; j++){
				w1[i][j] = w1[i][j] - alp * x[i] * a_back[j];
			}
		}
		for (int i = 0; i < HIDDEN + 1; i++){
			for (int j = 0; j < OUTPUT; j++){
				w2[i][j] = w2[i][j] - alp * a[i] * delta[j];
			}
		}

		errsum[ilearn] += error[isample]; /* Calculate the error(for log)*/

		/* Error rate */
		if (temp != tr_label[isample]) { err_cnt++; }
	}
}

/* Learning */
void learn(double lsum, double error[SAMPLE], double errsum[LEARN]){
	ofstream error_log1("error_val_log.txt"); //Log
	ofstream error_log2("error_loop_log.txt"); //Log

	cout << "L e a r n i n g  i s  S t a r t e d." << endl;

	for (int ilearn = 0; ilearn < LEARN; ilearn++){
		cout << "loop:" << ilearn << endl;
		train(ilearn, lsum, error, errsum);

		for (int m = 0; m < SAMPLE; m++){ error[m] = 0; } /* Initialaization */

		errsum[ilearn] /= (SAMPLE * 1.0);
		cout << "ERROR = " << errsum[ilearn] << endl;
		error_log1 << "loop :" << ilearn << " | ERROR = " << errsum[ilearn] << endl; //Log
		error_log2 << "loop : " << ilearn << " | error =" << err_cnt << " in " << SAMPLE << " Training Data" << endl; //Log
	}
	cout << "L e a r n i n g  i s  C o m p l e t e d." << endl;
}

/* Test */
void test(){
	int err = 0, temp;
	ofstream ofs4("test_result.txt"); //Log
	ofstream err_log("err_log.txt"); //Log
	double z1;

	cout << "T e s t  i s  S t a r t e d." << endl;

	for (int isample = 0; isample < SAMPLE2; isample++){
		/* Set the Inputs */
		for (int i = 0; i < INPUT; i++){
			x[i] = te_images[isample][i];
		}

		/* Threshold */
		x[INPUT] = (double)1.0;

		/* Calculate the Hiddens(a) */
		for (int j = 0; j < HIDDEN; j++){
			z1 = 0;
			for (int i = 0; i < INPUT + 1; i++){
				z1 = z1 + w1[i][j] * x[i];
			}
			a[j] = sigmoid(z1); /* Applying to Sigmoid Func */
		}

		/* Threshold */
		a[HIDDEN] = (double)1.0;

		/* Calculate the Outputs */
		for (int j = 0; j < OUTPUT; j++){
			z1 = 0;
			for (int i = 0; i < HIDDEN + 1; i++){
				z1 = z1 + w2[i][j] * a[i];
			}
			y[j] = sigmoid(z1); /* Applying to Sigmoid Func */
		}

		temp = 0;
		for (int j = 0; j < OUTPUT; j++){
			if (y[temp] < y[j]){
				temp = j;
			}
		}
		ofs4 << "sample:" << isample << " output = " << temp << "  ans = (" << te_label[isample] << ")" << endl; //Log
		ofs4 << " =>( y[0]=" << y[0] << " y[1]=" << y[1] << " y[2]=" << y[2] << " y[3]=" << y[3] << " y[4]=" << y[4] << " y[5]=" << y[5] << " y[6]=" << y[6] << " y[7]=" << y[7] << " y[8]=" << y[8] << " y[9]=" << y[9] << " )" << endl; //Log

		/* Error rate */
		if (temp != te_label[isample]) { err++; }

	}

	cout << "T e s t  i s  C o m p l e t e d ." << endl;
	cout << "Error = " << err << " in " << SAMPLE2 << " Test Data" << endl;
	err_log << "error = " << err << " in " << SAMPLE2 << " Test Data" << endl; //Log
}


int main(void){
	double alp;
	double lsum = 0.0;
	double error[SAMPLE] = {};
	double errsum[LEARN] = {};
	LARGE_INTEGER start_pc, end_pc, freq_pc; // for Timer
	double sec_pc; // for Timer

	Mnist mnist;
	mnist.readMnistData("MNIST/train-images.idx3-ubyte", "tr");
	mnist.readMnistLabel("MNIST/train-labels.idx1-ubyte", "tr");
	mnist.readMnistData("MNIST/t10k-images.idx3-ubyte", "te");
	mnist.readMnistLabel("MNIST/t10k-labels.idx1-ubyte", "te");

	setWeight1(w1);
	setWeight2(w2);

	cout << "S T A R T" << endl;

	QueryPerformanceFrequency(&freq_pc);
	QueryPerformanceCounter(&start_pc);

	learn(lsum, error, errsum);

	QueryPerformanceCounter(&end_pc);
	sec_pc = (end_pc.QuadPart - start_pc.QuadPart) / (double)freq_pc.QuadPart;

	test();

	printf("CPU :%.5f sec (only about learning)\n", sec_pc);
}